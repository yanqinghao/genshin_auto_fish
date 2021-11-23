import suanpan
from suanpan.app import app
from suanpan.app.arguments import String, List,Json
from suanpan.log import logger

@app.input(Json(key="inputData1", default="Suanpan"))
@app.output(Json(key="outputData1", alias="out1"))
@app.output(String(key="outputData2", alias="out2"))
def hello_world(context):
    args = context.args
    fish_list=args.inputData1
    if fish_list!=[]:
        logger.info(fish_list)
        return {"out1":fish_list}
    else:
        logger.info(fish_list)
        return {"out2":"cap again"}



if __name__ == "__main__":
    suanpan.run(app)
