from fastapi import FastAPI
from tasks import add, duplicate, execute_query_and_save_to_s3

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/add")
async def web_add(x: int, y: int, a: str):
    add.delay(x, y)
    return {"message": a}


@app.post("/duplicate")
async def web_duplicate(x: int, y: int, a: str):
    ans = duplicate.delay(y, x)
    print(ans)
    return {"message": a + " " + str(ans)}


@app.post("/execute_query_and_save_to_s3")
async def web_execute_query_and_save_to_s3(query: str):
    execute_query_and_save_to_s3.delay(query)
    return {"message": "great"}  # + str(ans)}


