from fastapi import FastAPI
import uvicorn

api = FastAPI()

@api.get("/")
def example_handler():
    return {"message": "Hello World!!!"}

if __name__ == "__main__":
    uvicorn.run(api)
