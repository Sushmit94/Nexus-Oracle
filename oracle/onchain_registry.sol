// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/security/Pausable.sol";

/**
 * @title PredictiveRouterOracle
 * @dev On-chain registry for miner health scores and routing eligibility
 * Provides trustless, auditable routing decisions for Cortensor network
 */
contract PredictiveRouterOracle is Ownable, ReentrancyGuard, Pausable {
    
    // ============ Structs ============
    
    struct MinerHealth {
        uint256 healthScore;        // 0-100 scaled to 0-10000 for precision
        uint256 failureProbability; // 0-100 scaled to 0-10000
        uint256 routingWeight;      // 0-100 scaled to 0-10000
        bool isEligible;            // Whether miner can receive traffic
        uint256 lastUpdated;        // Block timestamp of last update
        uint256 updateCount;        // Number of updates
        bytes32 metadataHash;       // Hash of off-chain metadata
    }
    
    struct OracleUpdate {
        address miner;
        uint256 healthScore;
        uint256 failureProbability;
        uint256 routingWeight;
        bool isEligible;
        uint256 timestamp;
        bytes32 evidenceHash;       // Hash of prediction evidence
    }
    
    struct AccessTier {
        uint256 queryFee;           // Fee per query in wei
        uint256 updateFee;          // Fee per update in wei
        uint256 subscriptionPrice;  // Monthly subscription in wei
        uint256 queryLimit;         // Max queries per period
    }
    
    // ============ State Variables ============
    
    mapping(address => MinerHealth) public minerHealth;
    mapping(address => bool) public registeredMiners;
    mapping(address => bool) public authorizedOracles;
    mapping(address => uint256) public subscriberExpiry;
    mapping(address => uint256) public queryCount;
    
    address[] public minerList;
    OracleUpdate[] public updateHistory;
    
    AccessTier public basicTier;
    AccessTier public premiumTier;
    
    uint256 public constant PRECISION = 10000;
    uint256 public constant MAX_HEALTH_SCORE = 10000;
    uint256 public constant ELIGIBILITY_THRESHOLD = 3000; // 30%
    
    uint256 public totalUpdates;
    uint256 public totalQueries;
    uint256 public oracleFeePool;
    
    // ============ Events ============
    
    event MinerRegistered(address indexed miner, uint256 timestamp);
    event MinerDeregistered(address indexed miner, uint256 timestamp);
    event HealthUpdated(
        address indexed miner,
        uint256 healthScore,
        uint256 failureProbability,
        uint256 routingWeight,
        bool isEligible,
        bytes32 evidenceHash
    );
    event OracleAuthorized(address indexed oracle, uint256 timestamp);
    event OracleRevoked(address indexed oracle, uint256 timestamp);
    event SubscriptionPurchased(address indexed subscriber, uint256 tier, uint256 expiry);
    event QueryExecuted(address indexed querier, address indexed miner, uint256 fee);
    event FeesWithdrawn(address indexed recipient, uint256 amount);
    event EmergencyReroute(address indexed miner, string reason);
    
    // ============ Modifiers ============
    
    modifier onlyAuthorizedOracle() {
        require(authorizedOracles[msg.sender], "Not authorized oracle");
        _;
    }
    
    modifier onlyRegisteredMiner(address miner) {
        require(registeredMiners[miner], "Miner not registered");
        _;
    }
    
    modifier hasQueryAccess() {
        require(
            subscriberExpiry[msg.sender] >= block.timestamp ||
            msg.value >= basicTier.queryFee,
            "Insufficient access"
        );
        _;
    }
    
    // ============ Constructor ============
    
    constructor() Ownable(msg.sender) {
        // Initialize access tiers
        basicTier = AccessTier({
            queryFee: 0.001 ether,
            updateFee: 0.01 ether,
            subscriptionPrice: 0.5 ether,
            queryLimit: 100
        });
        
        premiumTier = AccessTier({
            queryFee: 0,
            updateFee: 0.005 ether,
            subscriptionPrice: 2 ether,
            queryLimit: 10000
        });
        
        // Owner is initial authorized oracle
        authorizedOracles[msg.sender] = true;
    }
    
    // ============ Oracle Functions ============
    
    /**
     * @dev Register a new miner in the registry
     * @param miner Address of the miner
     * @param initialHealthScore Initial health score (0-10000)
     */
    function registerMiner(
        address miner,
        uint256 initialHealthScore
    ) external onlyAuthorizedOracle whenNotPaused {
        require(!registeredMiners[miner], "Already registered");
        require(initialHealthScore <= MAX_HEALTH_SCORE, "Invalid health score");
        
        registeredMiners[miner] = true;
        minerList.push(miner);
        
        minerHealth[miner] = MinerHealth({
            healthScore: initialHealthScore,
            failureProbability: 0,
            routingWeight: initialHealthScore,
            isEligible: initialHealthScore >= ELIGIBILITY_THRESHOLD,
            lastUpdated: block.timestamp,
            updateCount: 1,
            metadataHash: bytes32(0)
        });
        
        emit MinerRegistered(miner, block.timestamp);
    }
    
    /**
     * @dev Update miner health data
     * @param miner Address of the miner
     * @param healthScore New health score (0-10000)
     * @param failureProbability Predicted failure probability (0-10000)
     * @param routingWeight Recommended routing weight (0-10000)
     * @param evidenceHash Hash of prediction evidence
     */
    function updateMinerHealth(
        address miner,
        uint256 healthScore,
        uint256 failureProbability,
        uint256 routingWeight,
        bytes32 evidenceHash
    ) external onlyAuthorizedOracle onlyRegisteredMiner(miner) whenNotPaused {
        require(healthScore <= MAX_HEALTH_SCORE, "Invalid health score");
        require(failureProbability <= PRECISION, "Invalid probability");
        require(routingWeight <= PRECISION, "Invalid weight");
        
        MinerHealth storage health = minerHealth[miner];
        
        // Update health data
        health.healthScore = healthScore;
        health.failureProbability = failureProbability;
        health.routingWeight = routingWeight;
        health.isEligible = routingWeight >= ELIGIBILITY_THRESHOLD && failureProbability < 7000;
        health.lastUpdated = block.timestamp;
        health.updateCount++;
        
        // Record update in history
        updateHistory.push(OracleUpdate({
            miner: miner,
            healthScore: healthScore,
            failureProbability: failureProbability,
            routingWeight: routingWeight,
            isEligible: health.isEligible,
            timestamp: block.timestamp,
            evidenceHash: evidenceHash
        }));
        
        totalUpdates++;
        
        emit HealthUpdated(
            miner,
            healthScore,
            failureProbability,
            routingWeight,
            health.isEligible,
            evidenceHash
        );
    }
    
    /**
     * @dev Batch update multiple miners
     * @param miners Array of miner addresses
     * @param healthScores Array of health scores
     * @param failureProbabilities Array of failure probabilities
     * @param routingWeights Array of routing weights
     */
    function batchUpdateMiners(
        address[] calldata miners,
        uint256[] calldata healthScores,
        uint256[] calldata failureProbabilities,
        uint256[] calldata routingWeights
    ) external onlyAuthorizedOracle whenNotPaused {
        require(
            miners.length == healthScores.length &&
            miners.length == failureProbabilities.length &&
            miners.length == routingWeights.length,
            "Array length mismatch"
        );
        
        for (uint256 i = 0; i < miners.length; i++) {
            if (registeredMiners[miners[i]]) {
                MinerHealth storage health = minerHealth[miners[i]];
                health.healthScore = healthScores[i];
                health.failureProbability = failureProbabilities[i];
                health.routingWeight = routingWeights[i];
                health.isEligible = routingWeights[i] >= ELIGIBILITY_THRESHOLD && failureProbabilities[i] < 7000;
                health.lastUpdated = block.timestamp;
                health.updateCount++;
                
                emit HealthUpdated(
                    miners[i],
                    healthScores[i],
                    failureProbabilities[i],
                    routingWeights[i],
                    health.isEligible,
                    bytes32(0)
                );
            }
        }
        
        totalUpdates += miners.length;
    }
    
    /**
     * @dev Emergency reroute - immediately mark miner as ineligible
     * @param miner Address of the miner
     * @param reason Reason for emergency reroute
     */
    function emergencyReroute(
        address miner,
        string calldata reason
    ) external onlyAuthorizedOracle onlyRegisteredMiner(miner) {
        MinerHealth storage health = minerHealth[miner];
        
        health.routingWeight = 0;
        health.isEligible = false;
        health.failureProbability = PRECISION; // 100%
        health.lastUpdated = block.timestamp;
        health.updateCount++;
        
        emit EmergencyReroute(miner, reason);
        emit HealthUpdated(miner, health.healthScore, PRECISION, 0, false, bytes32(0));
    }
    
    // ============ Query Functions ============
    
    /**
     * @dev Query miner health (paid access)
     * @param miner Address of the miner
     */
    function queryMinerHealth(
        address miner
    ) external payable hasQueryAccess onlyRegisteredMiner(miner) returns (
        uint256 healthScore,
        uint256 failureProbability,
        uint256 routingWeight,
        bool isEligible,
        uint256 lastUpdated
    ) {
        // Collect fee if not subscriber
        if (subscriberExpiry[msg.sender] < block.timestamp) {
            require(msg.value >= basicTier.queryFee, "Insufficient fee");
            oracleFeePool += msg.value;
        }
        
        MinerHealth storage health = minerHealth[miner];
        
        queryCount[msg.sender]++;
        totalQueries++;
        
        emit QueryExecuted(msg.sender, miner, msg.value);
        
        return (
            health.healthScore,
            health.failureProbability,
            health.routingWeight,
            health.isEligible,
            health.lastUpdated
        );
    }
    
    /**
     * @dev Get all eligible miners for routing
     */
    function getEligibleMiners() external view returns (address[] memory) {
        uint256 eligibleCount = 0;
        
        // Count eligible miners
        for (uint256 i = 0; i < minerList.length; i++) {
            if (minerHealth[minerList[i]].isEligible) {
                eligibleCount++;
            }
        }
        
        // Build array
        address[] memory eligible = new address[](eligibleCount);
        uint256 index = 0;
        
        for (uint256 i = 0; i < minerList.length; i++) {
            if (minerHealth[minerList[i]].isEligible) {
                eligible[index] = minerList[i];
                index++;
            }
        }
        
        return eligible;
    }
    
    /**
     * @dev Get routing weights for all miners
     */
    function getRoutingWeights() external view returns (
        address[] memory miners,
        uint256[] memory weights
    ) {
        uint256 count = minerList.length;
        miners = new address[](count);
        weights = new uint256[](count);
        
        for (uint256 i = 0; i < count; i++) {
            miners[i] = minerList[i];
            weights[i] = minerHealth[minerList[i]].routingWeight;
        }
        
        return (miners, weights);
    }
    
    // ============ Subscription Functions ============
    
    /**
     * @dev Purchase subscription for query access
     * @param tier 0 for basic, 1 for premium
     */
    function purchaseSubscription(uint256 tier) external payable nonReentrant {
        AccessTier memory selectedTier = tier == 0 ? basicTier : premiumTier;
        require(msg.value >= selectedTier.subscriptionPrice, "Insufficient payment");
        
        // Extend or set subscription
        uint256 currentExpiry = subscriberExpiry[msg.sender];
        uint256 newExpiry = currentExpiry > block.timestamp 
            ? currentExpiry + 30 days 
            : block.timestamp + 30 days;
        
        subscriberExpiry[msg.sender] = newExpiry;
        oracleFeePool += msg.value;
        
        emit SubscriptionPurchased(msg.sender, tier, newExpiry);
    }
    
    // ============ Admin Functions ============
    
    /**
     * @dev Authorize a new oracle
     */
    function authorizeOracle(address oracle) external onlyOwner {
        authorizedOracles[oracle] = true;
        emit OracleAuthorized(oracle, block.timestamp);
    }
    
    /**
     * @dev Revoke oracle authorization
     */
    function revokeOracle(address oracle) external onlyOwner {
        authorizedOracles[oracle] = false;
        emit OracleRevoked(oracle, block.timestamp);
    }
    
    /**
     * @dev Update access tier pricing
     */
    function updateBasicTier(
        uint256 queryFee,
        uint256 updateFee,
        uint256 subscriptionPrice,
        uint256 queryLimit
    ) external onlyOwner {
        basicTier = AccessTier({
            queryFee: queryFee,
            updateFee: updateFee,
            subscriptionPrice: subscriptionPrice,
            queryLimit: queryLimit
        });
    }
    
    /**
     * @dev Update premium tier pricing
     */
    function updatePremiumTier(
        uint256 queryFee,
        uint256 updateFee,
        uint256 subscriptionPrice,
        uint256 queryLimit
    ) external onlyOwner {
        premiumTier = AccessTier({
            queryFee: queryFee,
            updateFee: updateFee,
            subscriptionPrice: subscriptionPrice,
            queryLimit: queryLimit
        });
    }
    
    /**
     * @dev Withdraw accumulated fees
     */
    function withdrawFees(address payable recipient) external onlyOwner nonReentrant {
        uint256 amount = oracleFeePool;
        oracleFeePool = 0;
        
        (bool success, ) = recipient.call{value: amount}("");
        require(success, "Transfer failed");
        
        emit FeesWithdrawn(recipient, amount);
    }
    
    /**
     * @dev Pause contract in emergency
     */
    function pause() external onlyOwner {
        _pause();
    }
    
    /**
     * @dev Unpause contract
     */
    function unpause() external onlyOwner {
        _unpause();
    }
    
    /**
     * @dev Deregister a miner
     */
    function deregisterMiner(address miner) external onlyOwner {
        require(registeredMiners[miner], "Not registered");
        registeredMiners[miner] = false;
        delete minerHealth[miner];
        
        emit MinerDeregistered(miner, block.timestamp);
    }
    
    // ============ View Functions ============
    
    /**
     * @dev Get total number of registered miners
     */
    function getMinerCount() external view returns (uint256) {
        return minerList.length;
    }
    
    /**
     * @dev Get recent update history
     */
    function getRecentUpdates(uint256 count) external view returns (OracleUpdate[] memory) {
        uint256 historyLength = updateHistory.length;
        uint256 resultCount = count > historyLength ? historyLength : count;
        
        OracleUpdate[] memory recent = new OracleUpdate[](resultCount);
        
        for (uint256 i = 0; i < resultCount; i++) {
            recent[i] = updateHistory[historyLength - 1 - i];
        }
        
        return recent;
    }
    
    /**
     * @dev Check if address is subscriber
     */
    function isSubscriber(address account) external view returns (bool) {
        return subscriberExpiry[account] >= block.timestamp;
    }
    
    /**
     * @dev Get subscription expiry for address
     */
    function getSubscriptionExpiry(address account) external view returns (uint256) {
        return subscriberExpiry[account];
    }
}