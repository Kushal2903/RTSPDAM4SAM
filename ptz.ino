/*
  3-AXIS GIMBAL (Roll / Pitch / Yaw)  -  ESP32-S3 + ICM-20948 + Kalman + ADRC|PID + CubeMars GL40 II (CAN)

  SAFE DEFAULT: USE_CAN 0  -> NOTHING is sent to motors. RollPWM/PitchPWM/YawPWM are VIRTUAL values
                for the Serial Plotter only (1000-2000, 1500 = zero command).
  Set USE_CAN 1 only after the bench test of the motors (one motor, unloaded, current-limited supply).

  Wiring: IMU SDA=GPIO8, SCL=GPIO9 (0x69, or 0x68 is auto-detected).
          CAN transceiver (SN65HVD230 / TJA1051T/3): ESP32 GPIO17 -> TXD, GPIO18 -> RXD,
          120 ohm at both ends of the bus, common ground. Motor CAN IDs 1,2,3 = roll,pitch,yaw.

  Serial 115200 commands:
    1 = ADRC mode    2 = PID mode    e = enable motors (needs USE_CAN 1)    x = disable motors
    z = clear target offsets    d = toggle motor-feedback debug
    R<deg> / P<deg> / Y<deg> = set target offset, e.g.  R5.0
*/
#include <Arduino.h>
#include <Wire.h>
#include "driver/twai.h"

// ======================= CONFIG =======================
#define USE_CAN      0   // 0 = virtual PWM only, 1 = real GL40 II over CAN
#define USE_MAG_YAW  0   // 0 = yaw from gyro only (drifts), 1 = magnetometer-corrected yaw (needs calibration)
#define OUT_MIT      1   // 1 = MIT torque mode (kp=kd=0, t_ff), 0 = speed mode

#define SDA_PIN 8
#define SCL_PIN 9
#define CAN_TX_PIN GPIO_NUM_17
#define CAN_RX_PIN GPIO_NUM_18

const uint8_t MOTOR_ID[3]     = {1, 2, 3};          // roll, pitch, yaw
const float   AXIS_SIGN[3]    = {1, 1, 1};          // flip to -1 per axis if it pushes the wrong way
const float   TORQUE_LIMIT[3] = {0.05f, 0.05f, 0.05f}; // N*m at u=1. START LOW. Rated ~0.25 N*m
const float   SPEED_LIMIT[3]  = {3.0f, 3.0f, 3.0f};    // speed-mode value at u=1 (sample uses rad/s-like units; verify)
const float   ERR_CUTOFF      = 60.0f;              // deg: beyond this the output is forced to 0

// ADRC (per axis): wc = controller bandwidth [rad/s], wo = observer bandwidth [rad/s] (~4-6x wc)
// B0 = deg/s^2 of camera acceleration per unit u  ~ (TORQUE_LIMIT / Inertia) * 57.3. TUNE THIS FIRST.
const float WC[3] = {20, 20, 15};
const float WO[3] = {100, 100, 75};
const float B0[3] = {5000, 5000, 5000};

// PID mode = your original controller (normalized by 90) + integral
const float PID_KP = 3.0f, PID_KI = 0.5f, PID_KD = 0.35f, PID_IMAX = 20.0f;

const float MAG_OFFSET[3] = {0, 0, 0};  // hard-iron offsets (uT) - calibrate if USE_MAG_YAW 1
const float MAG_SIGN = 1.0f;            // flip to -1 if yaw runs away with mag enabled

const float ACCEL_SCALE = 16384.0f;     // +-2 g
const float GYRO_SCALE  = 131.0f;       // +-250 deg/s

// ======================= GLOBALS =======================
float AccX, AccY, AccZ, GyroX, GyroY, GyroZ;
float RateRoll, RatePitch, RateYaw;
float AngleRoll, AnglePitch;
float biasR = 0, biasP = 0, biasY = 0;

float KalmanAngleRoll = 0.0f;
float KalmanUncertaintyAngleRoll = 1.0f;
float KalmanAnglePitch = 0.0f;
float KalmanUncertaintyAnglePitch = 1.0f;
float Kalman1DOutput[2] = {0.0f, 0.0f};

float YawAngle = 0, KalmanUncertaintyYaw = 1.0f;
float InitialRoll = 0, InitialPitch = 0, InitialYaw = 0;
float offsetDeg[3] = {0, 0, 0};
float gTgt[3] = {0, 0, 0}, uCtl[3] = {0, 0, 0}, pwmV[3] = {1500, 1500, 1500};

uint8_t imuAddr = 0x69, imuFail = 0, ctrlMode = 0;   // 0 = ADRC, 1 = PID
bool canRun = false, dbg = false;
int magDiv = 0;

// ======================= KALMAN (UNCHANGED STRUCTURE) =======================
void kalman_1d(
  float state,
  float uncertainty,
  float input,
  float measurement,
  float dt
)
{
  state = state + dt * input;

  const float gyroNoise = 4.0f;

  uncertainty =
    uncertainty +
    dt * dt *
    gyroNoise *
    gyroNoise;

  const float accelNoise = 2.0f;

  float KalmanGain =
    uncertainty /
    (
      uncertainty +
      accelNoise * accelNoise
    );

  state =
    state +
    KalmanGain *
    (
      measurement -
      state
    );

  uncertainty =
    (
      1.0f -
      KalmanGain
    ) *
    uncertainty;

  Kalman1DOutput[0] = state;
  Kalman1DOutput[1] = uncertainty;
}

// ======================= I2C / ICM-20948 =======================
bool wr(uint8_t a, uint8_t r, uint8_t v) {
  Wire.beginTransmission(a); Wire.write(r); Wire.write(v);
  return Wire.endTransmission() == 0;
}
bool rd(uint8_t a, uint8_t r, uint8_t *b, uint8_t n) {
  Wire.beginTransmission(a); Wire.write(r);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((uint16_t)a, (uint8_t)n, true) != n) return false;
  for (uint8_t i = 0; i < n; i++) b[i] = Wire.read();
  return true;
}

bool imuInit() {
  uint8_t cand[2] = {0x69, 0x68}, id = 0;
  bool found = false;
  for (int i = 0; i < 2 && !found; i++) {
    imuAddr = cand[i];
    wr(imuAddr, 0x7F, 0x00);                       // bank 0
    if (rd(imuAddr, 0x00, &id, 1) && id == 0xEA) found = true;
  }
  if (!found) return false;
  wr(imuAddr, 0x06, 0x80); delay(100);             // reset
  wr(imuAddr, 0x06, 0x01);                         // wake, auto clock
  wr(imuAddr, 0x07, 0x00);                         // enable accel + gyro
  wr(imuAddr, 0x7F, 0x20);                         // bank 2
  wr(imuAddr, 0x00, 0x00);                         // gyro rate div 0  (~1.1 kHz)
  wr(imuAddr, 0x01, 0x11);                         // gyro +-250 dps, DLPF cfg 2 (~120 Hz)
  wr(imuAddr, 0x10, 0x00); wr(imuAddr, 0x11, 0x00);// accel rate div 0 (~1.1 kHz)
  wr(imuAddr, 0x14, 0x11);                         // accel +-2 g, DLPF cfg 2
  wr(imuAddr, 0x7F, 0x00);                         // back to bank 0
#if USE_MAG_YAW
  wr(imuAddr, 0x0F, 0x02);                         // I2C bypass -> AK09916 visible at 0x0C
  wr(0x0C, 0x32, 0x01); delay(10);                 // mag soft reset
  wr(0x0C, 0x31, 0x08);                            // mag continuous 100 Hz
#endif
  return true;
}

bool readImu() {
  uint8_t b[14];
  if (!rd(imuAddr, 0x2D, b, 14)) return false;
  auto s = [&](int k) { return (float)(int16_t)((b[k] << 8) | b[k + 1]); };
  AccX = s(0) / ACCEL_SCALE; AccY = s(2) / ACCEL_SCALE; AccZ = s(4) / ACCEL_SCALE;
  GyroX = s(6); GyroY = s(8); GyroZ = s(10);       // raw LSB, divided by GYRO_SCALE later
  return true;
}

float wrap180(float a) { while (a > 180.0f) a -= 360.0f; while (a < -180.0f) a += 360.0f; return a; }

#if USE_MAG_YAW
bool readHeading(float &h) {                       // tilt-compensated heading, degrees
  uint8_t b[9];
  if (!rd(0x0C, 0x10, b, 9) || !(b[0] & 1)) return false;
  float mx0 = (int16_t)(b[1] | (b[2] << 8)) * 0.15f - MAG_OFFSET[0];
  float my0 = (int16_t)(b[3] | (b[4] << 8)) * 0.15f - MAG_OFFSET[1];
  float mz0 = (int16_t)(b[5] | (b[6] << 8)) * 0.15f - MAG_OFFSET[2];
  float mx = my0, my = mx0, mz = -mz0;             // align mag axes to accel/gyro frame (VERIFY on bench)
  float r = KalmanAngleRoll * PI / 180.0f, p = KalmanAnglePitch * PI / 180.0f;
  float xh = mx * cosf(p) + my * sinf(r) * sinf(p) + mz * cosf(r) * sinf(p);
  float yh = my * cosf(r) - mz * sinf(r);
  h = atan2f(-yh, xh) * 180.0f / PI;
  return true;
}
#endif

void imuStep(float dt) {
  if (!readImu()) { if (imuFail < 255) imuFail++; return; }
  imuFail = 0;

  RateRoll  = GyroX / GYRO_SCALE - biasR;
  RatePitch = GyroY / GYRO_SCALE - biasP;
  RateYaw   = GyroZ / GYRO_SCALE - biasY;

  AngleRoll =
    atan2f(
      AccY,
      sqrtf(
        AccX * AccX +
        AccZ * AccZ
      )
    ) *
    180.0f /
    PI;

  AnglePitch =
    -atan2f(
      AccX,
      sqrtf(
        AccY * AccY +
        AccZ * AccZ
      )
    ) *
    180.0f /
    PI;

  kalman_1d(KalmanAngleRoll, KalmanUncertaintyAngleRoll, RateRoll, AngleRoll, dt);
  KalmanAngleRoll = Kalman1DOutput[0];
  KalmanUncertaintyAngleRoll = Kalman1DOutput[1];

  kalman_1d(KalmanAnglePitch, KalmanUncertaintyAnglePitch, RatePitch, AnglePitch, dt);
  KalmanAnglePitch = Kalman1DOutput[0];
  KalmanUncertaintyAnglePitch = Kalman1DOutput[1];

  YawAngle += RateYaw * dt;                        // gyro prediction at 1 kHz
#if USE_MAG_YAW
  if (++magDiv >= 10) {                            // mag correction at ~100 Hz (input 0: prediction already applied)
    magDiv = 0;
    float h;
    if (readHeading(h)) {
      float meas = YawAngle + wrap180(MAG_SIGN * h - YawAngle);
      kalman_1d(YawAngle, KalmanUncertaintyYaw, 0.0f, meas, 0.01f);
      YawAngle = Kalman1DOutput[0];
      KalmanUncertaintyYaw = Kalman1DOutput[1];
    }
  }
#endif
}

// ======================= CAN (GL40 II, from official sample) =======================
#if USE_CAN
enum { M_MIT = 0, M_POSSPD = 1, M_SPD = 2 };
#define P_MIN -12.5f
#define P_MAX  12.5f
#define V_MIN -30.0f
#define V_MAX  30.0f
#define T_MIN -10.0f
#define T_MAX  10.0f
#define KP_MAX 500.0f
#define KD_MAX 5.0f
#if OUT_MIT
const uint8_t ACTIVE_MODE = M_MIT;
#else
const uint8_t ACTIVE_MODE = M_SPD;
#endif

void canSend(uint32_t id, const uint8_t *d, uint8_t len) {
  twai_message_t m = {};
  m.identifier = id; m.data_length_code = len; memcpy(m.data, d, len);
  twai_transmit(&m, 0);                            // non-blocking: never stall the control loop
}
void special(uint32_t id, uint8_t last) {
  uint8_t b[8] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, last};
  canSend(id, b, 8);
}
uint32_t f2u(float x, float lo, float hi, int bits) {
  x = constrain(x, lo, hi);
  return (uint32_t)((x - lo) * (((1UL << bits) - 1) / (hi - lo)));
}
float u2f(uint32_t u, float lo, float hi, int bits) {
  return (float)u * (hi - lo) / (float)((1UL << bits) - 1) + lo;
}
void cmdMIT(uint8_t id, float p, float v, float kp, float kd, float tff) {
  uint32_t pi = f2u(p, P_MIN, P_MAX, 16), vi = f2u(v, V_MIN, V_MAX, 12);
  uint32_t ki = f2u(kp, 0, KP_MAX, 12),   di = f2u(kd, 0, KD_MAX, 12), ti = f2u(tff, T_MIN, T_MAX, 12);
  uint8_t b[8] = { (uint8_t)(pi >> 8), (uint8_t)(pi & 0xFF), (uint8_t)(vi >> 4),
                   (uint8_t)(((vi & 0xF) << 4) | (ki >> 8)), (uint8_t)(ki & 0xFF), (uint8_t)(di >> 4),
                   (uint8_t)(((di & 0xF) << 4) | (ti >> 8)), (uint8_t)(ti & 0xFF) };
  canSend((M_MIT << 8) | id, b, 8);
}
void cmdSpd(uint8_t id, float spd) {
  uint8_t b[4]; memcpy(b, &spd, 4);
  canSend(id | (M_SPD << 8), b, 4);
}
void sendAxis(int i, float u) {
#if OUT_MIT
  cmdMIT(MOTOR_ID[i], 0, 0, 0, 0, u * TORQUE_LIMIT[i]);
#else
  cmdSpd(MOTOR_ID[i], u * SPEED_LIMIT[i]);
#endif
}
void motorsEnable() {
  for (int i = 0; i < 3; i++) { special((ACTIVE_MODE << 8) | MOTOR_ID[i], 0xFD); }
  delay(100);
  for (int i = 0; i < 3; i++) { special((ACTIVE_MODE << 8) | MOTOR_ID[i], 0xFC); }
  canRun = true;
}
void motorsDisable() {
  canRun = false;
  for (int i = 0; i < 3; i++) sendAxis(i, 0);
  delay(5);
  for (int i = 0; i < 3; i++) special((ACTIVE_MODE << 8) | MOTOR_ID[i], 0xFD);
}
void readFeedback() {
  twai_message_t m;
  while (twai_receive(&m, 0) == ESP_OK) {
    if (!dbg || m.data_length_code < 8) continue;
    float p = u2f((m.data[1] << 8) | m.data[2], P_MIN, P_MAX, 16);
    float v = u2f((m.data[3] << 4) | (m.data[4] >> 4), V_MIN, V_MAX, 12);
    float t = u2f(((m.data[4] & 0xF) << 8) | m.data[5], T_MIN, T_MAX, 12);
    Serial.printf("FB id:0x%X pos:%.3f spd:%.3f tq:%.3f Tdrv:%d Tmot:%d err:%u\n",
                  (unsigned)m.identifier, p, v, t, (int8_t)m.data[6], (int8_t)m.data[7], m.data[0] >> 4);
  }
}
#endif

// ======================= CONTROLLERS =======================
float z1[3], z2[3], z3[3], uPrev[3], pidI[3];
float gKp[3], gKd[3], gB1[3], gB2[3], gB3[3];

void computeGains() {
  for (int i = 0; i < 3; i++) {
    gKp[i] = WC[i] * WC[i];  gKd[i] = 2.0f * WC[i];
    gB1[i] = 3.0f * WO[i];   gB2[i] = 3.0f * WO[i] * WO[i];  gB3[i] = WO[i] * WO[i] * WO[i];
  }
}
void resetCtl(const float *y) {
  for (int i = 0; i < 3; i++) { z1[i] = y[i]; z2[i] = 0; z3[i] = 0; uPrev[i] = 0; pidI[i] = 0; }
}

// LADRC: plant  y'' = f + b0*u ;  ESO estimates y, y', f (total disturbance) ;  u = (kp*(r-z1) - kd*z2 - z3)/b0
float adrc(int i, float y, float tgt, float dt) {
  float e = z1[i] - y;
  z1[i] += dt * (z2[i] - gB1[i] * e);
  z2[i] += dt * (z3[i] + B0[i] * uPrev[i] - gB2[i] * e);
  z3[i] += dt * (-gB3[i] * e);
  float u0 = gKp[i] * (tgt - z1[i]) - gKd[i] * z2[i];
  float u = constrain((u0 - z3[i]) / B0[i], -1.0f, 1.0f);
  uPrev[i] = u;                                    // observer sees the SATURATED command (anti-windup)
  return u;
}

// PID mode: your original angle + rate-damping controller (normalized by 90) + integral
float pid(int i, float y, float tgt, float rate, float dt) {
  float e = tgt - y;
  pidI[i] = constrain(pidI[i] + e * dt, -PID_IMAX, PID_IMAX);
  float u = (PID_KP * e + PID_KI * pidI[i] - PID_KD * rate) / 90.0f;
  return constrain(u, -1.0f, 1.0f);
}

void controlStep(float dt) {
  float ang[3]  = {KalmanAngleRoll, KalmanAnglePitch, YawAngle};
  float rate[3] = {RateRoll, RatePitch, RateYaw};
  gTgt[0] = InitialRoll + offsetDeg[0];
  gTgt[1] = InitialPitch + offsetDeg[1];
  gTgt[2] = InitialYaw + offsetDeg[2];

  for (int i = 0; i < 3; i++) {
    float u;
    if (imuFail > 5 || fabsf(gTgt[i] - ang[i]) > ERR_CUTOFF) {   // failsafe: zero command
      u = 0; uPrev[i] = 0; z1[i] = ang[i]; z2[i] = 0; z3[i] = 0; pidI[i] = 0;
    } else {
      u = (ctrlMode == 0) ? adrc(i, ang[i], gTgt[i], dt) : pid(i, ang[i], gTgt[i], rate[i], dt);
    }
    uCtl[i] = u;
    pwmV[i] = 1500.0f + u * 500.0f;                // VIRTUAL PWM (plot only)
#if USE_CAN
    if (canRun) sendAxis(i, u * AXIS_SIGN[i]);     // REAL command to GL40 II
#endif
  }
}

// ======================= SERIAL COMMANDS =======================
void handleSerial() {
  while (Serial.available()) {
    char c = Serial.read();
    float ang[3] = {KalmanAngleRoll, KalmanAnglePitch, YawAngle};
    switch (c) {
      case '1': ctrlMode = 0; resetCtl(ang); Serial.println("MODE: ADRC"); break;
      case '2': ctrlMode = 1; resetCtl(ang); Serial.println("MODE: PID");  break;
      case 'z': offsetDeg[0] = offsetDeg[1] = offsetDeg[2] = 0; break;
      case 'd': dbg = !dbg; break;
      case 'R': offsetDeg[0] = Serial.parseFloat(); break;
      case 'P': offsetDeg[1] = Serial.parseFloat(); break;
      case 'Y': offsetDeg[2] = Serial.parseFloat(); break;
      case 'e':
#if USE_CAN
        resetCtl(ang); motorsEnable(); Serial.println("MOTORS ENABLED");
#else
        Serial.println("USE_CAN is 0: virtual PWM only");
#endif
        break;
      case 'x':
#if USE_CAN
        motorsDisable(); Serial.println("MOTORS DISABLED");
#endif
        break;
    }
  }
}

// ======================= SETUP / LOOP =======================
uint32_t lastImuUs,
