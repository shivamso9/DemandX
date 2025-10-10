import logging
import subprocess
import sys

logger = logging.getLogger("o9_logger")


def upgrade_package(package_name, version):
    try:
        import_module = __import__(package_name)
        logger.info(f"{package_name} version: {import_module.__version__}")

        logger.info(f"--- Installing {package_name} == {version} --- ")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", f"{package_name}=={version}"],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info("Installation successful.")

        # Re-import the package to confirm the version
        import_module = __import__(package_name)
        logger.info(f"{package_name} version: {import_module.__version__}")

    except subprocess.CalledProcessError as e:
        logger.error("Installation failed.")
        logger.error("Error details:")
        logger.error(e.stderr)
