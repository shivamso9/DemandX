# run.py

from pluginfactoryapp import create_app
import os

app = create_app()

if __name__ == '__main__':
    # Use environment variables for port, with a default for local development
    port = int(os.environ.get('PORT', 8080))
    # debug=True is great for development, but should be False in production
    app.run(host='0.0.0.0', port=port, debug=True)
    