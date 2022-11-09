from fastapi import FastAPI
import uvicorn

api = FastAPI()

@api.get("/")
def example_handler():
    return {"message": "Hello Wo!!!"}

if __name__ == "__main__":
    uvicorn(api)
