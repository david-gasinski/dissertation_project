from dotenv import load_dotenv

from app import create_app, limiter

app = create_app()

if __name__ == '__main__':
    print("Loading environment variables...")
    load_dotenv()
    
    print("Started backend.")
    # run app and initialise limiter
    limiter.init_app(app)
    app.run()