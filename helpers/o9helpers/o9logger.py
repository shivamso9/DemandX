import logging
import sys


class O9Logger:
    """
    A logger class that gets the 'o9_logger' instance.
    """

    def __init__(self):
        """Initializes by getting the logger instance."""
        self.logger = logging.getLogger("o9_logger")

    def info(self, message: str):
        """Logs a message with the INFO level."""
        self.logger.info(message)

    def warning(self, message: str):
        """Logs a message with the WARNING level."""
        self.logger.warning(message)

    def debug(self, message: str):
        """Logs a message with the DEBUG level."""
        self.logger.debug(message)

    def exception(self, message: str):
        """
        Logs a message with the EXCEPTION level.
        This method automatically includes exception information when called
        from within an 'except' block.
        """
        self.logger.exception(message)


# --- Example Usage: How a main application would use this class ---
if __name__ == "__main__":

    # --- Step 1: Configure the logger (done once in the main application) ---
    # This is the crucial step. We get the logger by name and add a handler to it.
    # Any instance of O9Logger will now send logs to this handler.
    print("--- Configuring logger in main application ---")
    log_instance_for_config = logging.getLogger("o9_logger")
    log_instance_for_config.setLevel(logging.DEBUG)  # Set the lowest level to capture all messages.

    handler = logging.StreamHandler(sys.stdout)  # Handler to print to console
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)

    # Add the configured handler to the logger
    if not log_instance_for_config.handlers:
        log_instance_for_config.addHandler(handler)

    print("--- Logger configured. Running demo. ---\n")

    # --- Step 2: Use the O9Logger class ---
    log = O9Logger()
    log.info("Application has started.")
    log.debug("This is a detailed debug message.")
