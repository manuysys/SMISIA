from fastapi import FastAPI
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime
import time
import threading
import random

def auto_generate_data():
    while True:
        db = SessionLocal()
        
        new_data = SensorReading(
            silo_id="A12",
            temperature=random.uniform(20, 35),
            humidity=random.uniform(10, 25),
            co2=random.randint(1000, 1600),
            presion=random.randint(900, 1100),
        )

        db.add(new_data)
        db.commit()
        db.close()

        print("Dato guardado automáticamente")

        time.sleep(120)  #cambiar cada cuanto acutaliza los datos

app = FastAPI()

DATABASE_URL = "sqlite:///./sensores.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)

Base = declarative_base()

class SensorReading(Base):
    __tablename__ = "sensor_readings"

    id = Column(Integer, primary_key=True, index=True)
    silo_id = Column(String)
    temperature = Column(Float)
    humidity = Column(Float)
    co2 = Column(Float)
    presion = Column(Float)
    timestamp = Column(String, default=lambda: datetime.now().isoformat())


Base.metadata.create_all(bind=engine)


@app.get("/")
def home():
    return {"message": "SMISIA backend con DB funcionando"}


@app.post("/sensor-data")
def receive_sensor_data(data: dict):
    db = SessionLocal()

    new_data = SensorReading(
        silo_id=data.get("silo_id"),
        temperature=data.get("temperature"),
        humidity=data.get("humidity"),
        co2=data.get("co2"),
        presion=data.get("presion"),
    )

    db.add(new_data)
    db.commit()
    db.close()

    return {"status": "saved in database"}

# leer datos
@app.get("/sensor-data")
def get_data():
    db = SessionLocal()
    data = db.query(SensorReading).all()
    db.close()

    return [
        {
            "id": d.id,
            "silo_id": d.silo_id,
            "temperature": d.temperature,
            "humidity": d.humidity,
            "co2": d.co2,
            "presion": d.presion,
            "timestamp": d.timestamp
        }
        for d in data
    ]

@app.get("/silos/{silo_id}/status")
def get_silo_status(silo_id: str):
    db = SessionLocal()

    
    data = (
        db.query(SensorReading)
        .filter(SensorReading.silo_id == silo_id)
        .order_by(SensorReading.id.desc())
        .first()
    )

    db.close()

    if not data:
        return {"error": "No hay datos para este silo"}

    if data.temperature > 30 or data.humidity > 20:
        status = "alerta"
    else:
        status = "bien"

    return {
        "status": status,
        "temperature": data.temperature,
        "humidity": data.humidity
    }

@app.post("/predict")
def predict(data: dict):
    temperature = data.get("temperature")
    humidity = data.get("humidity")
    co2 = data.get("co2")
    presion = data.get("presion")

    if temperature > 35 or humidity > 20 or co2 > 1400 or presion > 1000:
        status = "riesgo"
    else:
        status = "normal"

    return {"status": status}

@app.on_event("startup")
def start_auto_data():
    thread = threading.Thread(target=auto_generate_data)
    thread.daemon = True
    thread.start()