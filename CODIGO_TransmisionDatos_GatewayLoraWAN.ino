#include "LoRaWan_APP.h"
#include "Arduino.h"

#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BME280.h>
#include <Adafruit_SHT31.h>

// ================== LoRaWAN ==================
uint8_t devEui[] = { 0x70, 0xB3, 0xD5, 0x7E, 0xD0, 0x07, 0x6D, 0xC9 };
uint8_t appEui[] = { 0xA2, 0xF4, 0x2E, 0x5B, 0xA3, 0x9F, 0x0F, 0x21 };
uint8_t appKey[] = { 0xDF, 0x89, 0xBD, 0xB2, 0xA5, 0x64, 0x92, 0xCF, 0x23, 0x17, 0x1D, 0x4A, 0xEC, 0xB8, 0xEA, 0x5F };

LoRaMacRegion_t loraWanRegion = LORAMAC_REGION_AU915;
DeviceClass_t loraWanClass = CLASS_A;
bool overTheAirActivation = true;
bool loraWanAdr = true;
bool isTxConfirmed = false;
uint8_t appPort = 2;

uint32_t appTxDutyCycle = 5000; // 1 hora

// ================== Sensores ==================
Adafruit_BME280 bme;
Adafruit_SHT31 sht30 = Adafruit_SHT31();

const float UMBRAL_MOVIMIENTO = 18.0; 

unsigned long tiempoAnterior = 0;
const long intervaloClima = 2000;

bool bme_ok = true;

// ================== FUNCIONES ==================

void checkBME(float t, float h, float p)
{
  if (t == 0 && h == 0 && p == 0) {
    bme_ok = false;
  }
}

bool iniciarBME() {
  for (int i = 0; i < 5; i++) {
    if (bme.begin(0x76) || bme.begin(0x77)) {
      return true;
    }
    delay(500);
  }
  return false;
}

// 🔥 ESTA FUNCIÓN MANDA LOS DATOS A TTN
void prepareTxFrame(uint8_t port)
{
  float temp = sht30.readTemperature();
  float hum = sht30.readHumidity();
  float pres = bme.readPressure() / 100.0F;

  int t = temp * 100;
  int h = hum * 100;
  int p = pres;

  appDataSize = 6;

  appData[0] = highByte(t);
  appData[1] = lowByte(t);
  appData[2] = highByte(h);
  appData[3] = lowByte(h);
  appData[4] = highByte(p);
  appData[5] = lowByte(p);

  Serial.println("📡 Datos preparados para enviar");
}

void setup() {
  Serial.begin(115200);
  delay(2000);

  pinMode(Vext, OUTPUT);
  digitalWrite(Vext, LOW);

  Wire.begin();
  Wire.setClock(50000);

  iniciarBME();
  sht30.begin(0x44);

  // MPU6050 init
  Wire.beginTransmission(0x68);
  Wire.write(0x6B);
  Wire.write(0x00);
  Wire.endTransmission(true);

  // 🔥 INIT LORA
  deviceState = DEVICE_STATE_INIT;
}

void loop() {

  // ================= MPU =================
  Wire.beginTransmission(0x68);
  Wire.write(0x3B);
  Wire.endTransmission(false);
  Wire.requestFrom((uint16_t)0x68, (size_t)6, true);

  if (Wire.available() >= 6) {
    int16_t ax_raw = Wire.read() << 8 | Wire.read();
    int16_t ay_raw = Wire.read() << 8 | Wire.read();
    int16_t az_raw = Wire.read() << 8 | Wire.read();

    float ax = (ax_raw / 8192.0) * 9.81;
    float ay = (ay_raw / 8192.0) * 9.81;
    float az = (az_raw / 8192.0) * 9.81;

    float magnitud = sqrt(ax*ax + ay*ay + az*az);

    if (magnitud > UMBRAL_MOVIMIENTO) {
      Serial.println("🚨 MOVIMIENTO 🚨");
    }
  }

  // ================= LoRaWAN =================
  switch (deviceState)
  {
    case DEVICE_STATE_INIT:
      LoRaWAN.init(loraWanClass, loraWanRegion);
      break;

    case DEVICE_STATE_JOIN:
      LoRaWAN.join();
      break;

    case DEVICE_STATE_SEND:
      prepareTxFrame(appPort);
      LoRaWAN.send();
      deviceState = DEVICE_STATE_CYCLE;
      break;

    case DEVICE_STATE_CYCLE:
      txDutyCycleTime = appTxDutyCycle;
      LoRaWAN.cycle(txDutyCycleTime);
      deviceState = DEVICE_STATE_SLEEP;
      break;

    case DEVICE_STATE_SLEEP:
      LoRaWAN.sleep();
      break;

    default:
      deviceState = DEVICE_STATE_INIT;
      break;
  }
}