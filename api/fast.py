import pandas as pd
import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# write some code for the API here
@app.get("/")
def index():
    return {"greeting": "Hello world"}


@app.get("/predict_fare/")
def create_fare(
                pickup_datetime,
                pickup_longitude,
                pickup_latitude,
                dropoff_longitude,
                dropoff_latitude,
                passenger_count):
    # key = "2013-07-06 17:18:00.000000119"
    # pickup_datetime = "2013-07-06 17:18:00 UTC"
    # pickup_longitude = "-73.950655"
    # pickup_latitude = "40.783282"
    # dropoff_longitude = "-73.984365"
    # dropoff_latitude = "40.769802"
    # passenger_count = "1"
    # build X :warning: beware to the order of the parameters :warning:
    X = pd.DataFrame(dict(
        key=0,
        pickup_datetime=[pickup_datetime],
        pickup_longitude=[float(pickup_longitude)],
        pickup_latitude=[float(pickup_latitude)],
        dropoff_longitude=[float(dropoff_longitude)],
        dropoff_latitude=[float(dropoff_latitude)],
        passenger_count=[int(passenger_count)]))
    # :warning: TODO: get model from GCP
    # pipeline = get_model_from_gcp()
    pipeline = joblib.load('model.joblib')
    # make prediction
    results = pipeline.predict(X)
    # convert response from numpy to python type
    pred = float(results[0])
    return dict(
        prediction=pred)
