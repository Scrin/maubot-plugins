"""Configuration for the ChatGPT bot."""

from mautrix.util.config import BaseProxyConfig, ConfigUpdateHelper

class Config(BaseProxyConfig):
    """Configuration class for the ChatGPT bot."""

    def do_update(self, helper: ConfigUpdateHelper) -> None:
        """Update the configuration with new values.
        
        Args:
            helper: Helper class for updating config values
        """
        helper.copy("api-key")
        helper.copy("bot-name")
        helper.copy("model")
        helper.copy("allowed_models")
        helper.copy("vat")
        helper.copy("api-endpoint") 
