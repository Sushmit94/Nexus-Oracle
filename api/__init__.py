# API Module
# Paid Prediction API with authentication

from .prediction_api import app as prediction_api
from .auth import AuthManager

__all__ = ["prediction_api", "AuthManager"]