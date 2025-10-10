class O9Exception(Exception):
    """
    Base class for custom exceptions in this project.
    It allows for a custom message and an optional error code.
    """

    def __init__(self, message: str, error_code: int = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

    def __str__(self):
        """Returns a formatted string representation of the exception."""
        if self.error_code:
            return f"[Error {self.error_code}] {self.message}"
        return self.message


class ConfigurationError(O9Exception):
    """Raised for errors in a configuration file or value."""

    pass


class PluginException(O9Exception):
    """Raised when a plugin fails to execute correctly."""

    pass


class DataProcessingError(O9Exception):
    """Raised for errors during data transformation or processing."""

    pass


class InputEmptyException(O9Exception):
    """Custom exception for handling empty inputs."""

    pass


class O9ValueError(O9Exception):
    """Raised when a value is inappropriate for the operation."""

    pass


class O9KeyError(O9Exception):
    """Raised when a dictionary key is not found."""

    pass


class o9RuntimeException(O9Exception):
    """Raised for runtime errors that are not covered by other exceptions."""

    pass


class O9TypeError(O9Exception):
    """Raised when an operation or function is applied to an object of inappropriate type."""

    pass


class O9NotImplementedError(O9Exception):
    """Raised when a method or function is not implemented."""

    pass


class O9TimeoutError(O9Exception):
    """Raised when an operation exceeds the allowed time limit."""

    pass


class O9ConnectionError(O9Exception):
    """Raised when a connection to a service or resource fails."""

    pass


class AssertionError(O9Exception):
    """Raised when an assertion fails."""

    pass


class AttributeError(O9Exception):
    """Raised when an attribute reference or assignment fails."""

    pass


class O9ImportError(O9Exception):
    """Raised when an import statement fails."""

    pass


class SchemaCreationFailedException(O9Exception):
    """Raised when schema creation fails."""

    pass


# --- Example of how to use the improved exceptions ---
if __name__ == "__main__":
    # Example 1: Raising with just a message
    try:
        raise PluginException("The plugin failed to load.")
    except PluginException as e:
        print(f"Caught exception: {e}")

    # Example 2: Raising with a message and an error code
    try:
        raise ConfigurationError("API key is missing.", error_code=1001)
    except ConfigurationError as e:
        print(f"Caught exception: {e}")
