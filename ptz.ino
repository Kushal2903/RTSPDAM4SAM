/*
  3-AXIS GIMBAL  -  ESP32-S3 + ICM-20948 + Kalman(exact) + CubeMars GL40 II (CAN)

  MODE 1  ADRC = EARTH STABILIZATION. Camera attitude held in the world frame:
                 roll/pitch vs gravity, yaw vs magnetic heading. Feedback = IMU.
  MODE 2  PID  = PTZ COMMAND/POINTING. Motors go to commanded angles and hold.
                 Feedback = motor encoder from the GL40 II over CAN (IMU not used for control).
  Modes are separate; only one runs at a time.

  SAFE DEFAULT: USE_CAN 0 -> nothing is sent to motors. RollPWM/PitchPWM/YawPWM are VIRTUAL
  (1000-2000, 1500 = zero command). In PID mode a simple virtual motor model stands in for the encoder.

  Wiring: IMU SDA=GPIO8, SCL=GPIO9 (0x69, or 0x68 auto-detected).
          CAN transceiver (SN65HVD230 / TJA1051T/3): GPIO17 -> TXD, GPIO18 -> RXD, 120 ohm at both bus ends.
          Motor CAN IDs: 1 = roll, 2 = pitch, 3 = yaw.

  Serial Monitor 115200, set line ending to "Newline". Commands:
    1        ADRC mode (locks the CURRENT view as the world-frame target)
    2        PID mode (holds the current motor angle)
    t        re-lock current view as target (ADRC)
    R5.0     roll target:  ADRC = world offset (deg),  PID = commanded motor angle (deg)
    P-10     pitch target (same meaning)        Y30   yaw target (same meaning)
    z        zero all offsets / commands
    e / x    enable / disable motors (needs USE_CAN 1)
    o        zero motor encoders here (PID reference)
    m        magnetometer hard-iron calibration (15 s, rotate the IMU through all orientations, motors off)
    d        toggle debug (motor feedback + gyro-yaw vs mag-heading)
*/
#include <Arduino.h>
#include <Wire.h>
#include "driver/twai.h"

// ======================= CONFIG =======================
#define USE_CAN      0   // 0 = virtual PWM only, 1 = real GL40 II over CAN
#define USE_MAG_YAW  1   // 1 = yaw referenced to magnetic heading (earth direction). 0 = gyro-only (drifts)
#define OUT_MIT      1   // 1 = MIT torque mode (kp=kd=0, t_ff), 0 = speed mode

#define SDA_PIN 8
#define SCL_PIN 9
#define CAN_TX_PIN GPIO_NUM_17
#define CAN_RX_PIN GPIO_NUM_18

const uint8_t MOTOR_ID[3]     = {1, 2, 3};             // roll, pitch, yaw
const float   AXIS_SIGN[3]    = {1, 1, 1};             // ADRC only: flip to -1 if an axis pushes the wrong way
const float   TORQUE_LIMIT[3] = {0.05f, 0.05f, 0.05f}; // N*m at u=1. START LOW (rated ~0.25 N*m)
const float   SPEED_LIMIT[3]  = {3.0f, 3.0f, 3.0f};    // speed-mode value at u=1 (verify units)
const float   ERR_CUTOFF      = 60.0f;                 // deg: larger error -> output forced to 0
const float   SLEW_DPS[3]     = {60, 60, 90};          // max target slew (deg/s), both modes -> smooth moves
const float   CMD_LIMIT[3]    = {45, 60, 170};         // PID mode: command limits +- deg

// ADRC (per axis): WC = controller bandwidth [rad/s], WO = observer bandwidth (~4-6x WC)
// B0 = deg/s^2 of camera acceleration per unit u ~ (TORQUE_LIMIT / Inertia) * 57.3. TUNE B0 FIRST.
const float WC[3] = {20, 20, 15};
const float WO[3] = {100, 100, 75};
const float B0[3] = {5000, 5000, 5000};

// PID (encoder domain, degrees, normalized by 90 like your original controller)
const float PID_KP = 3.0f, PID_KI = 0.3f, PID_KD = 0.35f, PID_IMAX = 20.0f;

// Virtual motor model (USE_CAN 0, PID mode only)
const float VM_ACC = 800.0f, VM_DAMP = 6.0f;

const float MAG_SIGN = 1.0f;            // flip to -1 if yaw from mag runs opposite to gyro yaw (see debug 'd')
const float ACCEL_SCALE = 16384.0f;     // +-2 g
const float GYRO_SCALE  = 131.0f;       // +-250 deg/s
const float RAD2DEG = 57.29578f;

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
float InitialRoll = 0, InitialPitch = 0, InitialYaw = 0;   // world-frame stabilization targets
float offsetDeg[3] = {0, 0, 0};     // ADRC: world offsets
float cmdDeg[3]    = {0, 0, 0};     // PID: commanded motor angles
float ramp[3]      = {0, 0, 0};     // slew-limited active target
float gTgt[3] = {0, 0, 0}, uCtl[3] = {0, 0, 0}, pwmV[3] = {1500, 1500, 1500};
float angOut[3] = {0, 0, 0};        // angles shown in plot (camera world angles or motor angles)

uint8_t imuAddr = 0x69, imuFail = 0, ctrlMode = 0;   // 0 = ADRC, 1 = PID
bool canRun = false, dbg = false;
int magDiv = 0;
float magOff[3] = {0, 0, 0}, magMin[3], magMax[3], lastHeading = 0;
uint32_t magCalEnd = 0;

// motor feedback (CAN) and virtual motor
float fbPosDeg[3] = {0, 0, 0}, fbSpdDps[3] = {0, 0, 0}, motZero[3] = {0, 0, 0};
uint32_t fbMs[3] = {0, 0, 0};
bool needZero[3] = {true, true, true};
float vmPos[3] = {0, 0, 0}, vmSpd[3] = {0, 0, 0};

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
    wr(imuAddr, 0x7F, 0x00);                        // bank 0
    if (rd(imuAddr, 0x00, &id, 1) && id == 0xEA) found = true;
  }
  if (!found) return false;
  wr(imuAddr, 0x06, 0x80); delay(100);              // reset
  wr(imuAddr, 0x06, 0x01);                          // wake, auto clock
  wr(imuAddr, 0x07, 0x00);                          // accel + gyro on
  wr(imuAddr, 0x7F, 0x20);                          // bank 2
  wr(imuAddr, 0x00, 0x00);                          // gyro rate div 0 (~1.1 kHz)
  wr(imuAddr, 0x01, 0x11);                          // gyro +-250 dps, DLPF 2
  wr(imuAddr, 0x10, 0x00); wr(imuAddr, 0x11, 0x00); // accel rate div 0
  wr(imuAddr, 0x14, 0x11);                          // accel +-2 g, DLPF 2
  wr(imuAddr, 0x7F, 0x00);                          // bank 0
#if USE_MAG_YAW
  wr(imuAddr, 0x0F, 0x02);                          // I2C bypass: AK09916 visible at 0x0C
  wr(0x0C, 0x32, 0x01); delay(10);                  // mag soft reset
  wr(0x0C, 0x31, 0x08);                             // mag continuous 100 Hz
#endif
  return true;
}

bool readImu() {
  uint8_t b[14];
  if (!rd(imuAddr, 0x2D, b, 14)) return false;
  auto s = [&](int k) { return (float)(int16_t)((b[k] << 8) | b[k + 1]); };
  AccX = s(0) / ACCEL_SCALE; AccY = s(2) / ACCEL_SCALE; AccZ = s(4) / ACCEL_SCALE;
  GyroX = s(6); GyroY = s(8); GyroZ = s(10);        // raw LSB; divided by GYRO_SCALE later
  return true;
}

float wrap180(float a) { while (a > 180.0f) a -= 360.0f; while (a < -180.0f) a += 360.0f; return a; }

#if USE_MAG_YAW
bool readMagRaw(float *m) {                          // uT in sensor axes
  uint8_t b[9];
  if (!rd(0x0C, 0x10, b, 9) || !(b[0] & 1)) return false;
  m[0] = (int16_t)(b[1] | (b[2] << 8)) * 0.15f;
  m[1] = (int16_t)(b[3] | (b[4] << 8)) * 0.15f;
  m[2] = (int16_t)(b[5] | (b[6] << 8)) * 0.15f;
  return true;
}
float heading(const float *raw) {                    // tilt-compensated, CCW-positive like gyro Z (Z up)
  float mx0 = raw[0] - magOff[0], my0 = raw[1] - magOff[1], mz0 = raw[2] - magOff[2];
  float mx = my0, my = mx0, mz = -mz0;               // AK09916 axes -> accel/gyro axes (VERIFY on bench)
  float r = KalmanAngleRoll * PI / 180.0f, p = KalmanAnglePitch * PI / 180.0f;
  float xh = mx * cosf(p) + my * sinf(r) * sinf(p) + mz * cosf(r) * sinf(p);
  float yh = my * cosf(r) - mz * sinf(r);
  return MAG_SIGN * atan2f(-yh, xh) * 180.0f / PI;
}
void finishMagCal() {
  for (int k = 0; k < 3; k++) magOff[k] = 0.5f * (magMin[k] + magMax[k]);
  magCalEnd = 0;
  Serial.printf("MAG CAL DONE. magOff = {%.2f, %.2f, %.2f}  (paste into magOff[] to keep). Press 1 to re-lock heading.\n",
                magOff[0], magOff[1], magOff[2]);
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

  YawAngle += RateYaw * dt;                          // gyro prediction @1 kHz
#if USE_MAG_YAW
  if (++magDiv >= 10) {                              // mag @ ~100 Hz
    magDiv = 0;
    float raw[3];
    if (readMagRaw(raw)) {
      if (magCalEnd) {                               // calibration: track min/max, no fusion
        for (int k = 0; k < 3; k++) { magMin[k] = min(magMin[k], raw[k]); magMax[k] = max(magMax[k], raw[k]); }
        if (millis() > magCalEnd) finishMagCal();
      } else {
        float h = heading(raw);
        lastHeading = h;
        float meas = YawAngle + wrap180(h - YawAngle);   // continuous (no +-180 jump)
        kalman_1d(YawAngle, KalmanUncertaintyYaw, 0.0f, meas, 0.01f);
        YawAngle = Kalman1DOutput[0];
        KalmanUncertaintyYaw = Kalman1DOutput[1];
      }
    }
  }
#endif
}

// ======================= CAN (GL40 II, from official sample) =======================
float motorDeg(int i) {
#if USE_CAN
  return fbPosDeg[i] - motZero[i];
#else
  return vmPos[i];
#endif
}
float motorSpd(int i) {
#if USE_CAN
  return fbSpdDps[i];
#else
  return vmSpd[i];
#endif
}
bool motorValid(int i) {
#if USE_CAN
  return canRun && !needZero[i] && (millis() - fbMs[i] < 50);
#else
  return true;
#endif
}

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
  twai_transmit(&m, 0);                              // non-blocking: never stall the loop
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
  for (int i = 0; i < 3; i++) { special((ACTIVE_MODE << 8) | MOTOR_ID[i], 0xFD); needZero[i] = true; fbMs[i] = 0; }
  delay(100);
  for (int i = 0; i < 3; i++) special((ACTIVE_MODE << 8) | MOTOR_ID[i], 0xFC);
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
    if (m.data_length_code < 8) continue;
    // Which motor replied? Official sample doesn't say. Try CAN-ID low byte, then low nibble of data[0].
    int idx = -1;
    for (int i = 0; i < 3; i++) if ((m.identifier & 0xFF) == MOTOR_ID[i]) idx = i;
    if (idx < 0) for (int i = 0; i < 3; i++) if ((m.data[0] & 0x0F) == MOTOR_ID[i]) idx = i;
    float p = u2f((m.data[1] << 8) | m.data[2], P_MIN, P_MAX, 16);
    float v = u2f((m.data[3] << 4) | (m.data[4] >> 4), V_MIN, V_MAX, 12);
    float t = u2f(((m.data[4] & 0xF) << 8) | m.data[5], T_MIN, T_MAX, 12);
    if (dbg) Serial.printf("FB id:0x%X idx:%d pos:%.3f spd:%.3f tq:%.3f Tdrv:%d Tmot:%d err:%u\n",
                           (unsigned)m.identifier, idx, p, v, t, (int8_t)m.data[6], (int8_t)m.data[7], m.data[0] >> 4);
    if (idx < 0) continue;
    fbPosDeg[idx] = p * RAD2DEG; fbSpdDps[idx] = v * RAD2DEG; fbMs[idx] = millis();
    if (needZero[idx]) { motZero[idx] = fbPosDeg[idx]; needZero[idx] = false; }
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
void resetCtl() {
  float y[3] = {KalmanAngleRoll, KalmanAnglePitch, YawAngle};
  for (int i = 0; i < 3; i++) { z1[i] = y[i]; z2[i] = 0; z3[i] = 0; uPrev[i] = 0; pidI[i] = 0; }
}

// LADRC: y'' = f + b*u. ESO estimates y, y', f (total disturbance). u = (kp*(r-z1) - kd*z2 - z3)/b
// AXIS_SIGN is folded into b so the observer matches the real direction of the motor.
float adrc(int i, float y, float tgt, float dt) {
  float b = B0[i] * AXIS_SIGN[i];
  float e = z1[i] - y;
  z1[i] += dt * (z2[i] - gB1[i] * e);
  z2[i] += dt * (z3[i] + b * uPrev[i] - gB2[i] * e);
  z3[i] += dt * (-gB3[i] * e);
  float u0 = gKp[i] * (tgt - z1[i]) - gKd[i] * z2[i];
  float u = constrain((u0 - z3[i]) / b, -1.0f, 1.0f);
  uPrev[i] = u;                                      // observer sees the SATURATED command
  return u;
}

// PID on motor encoder angle (deg). D term uses measured speed (no derivative kick on target steps).
float pid(int i, float y, float spd, float tgt, float dt) {
  float e = tgt - y;
  float raw = (PID_KP * e + PID_KI * pidI[i] - PID_KD * spd) / 90.0f;
  if (fabsf(raw) < 1.0f) pidI[i] = constrain(pidI[i] + e * dt, -PID_IMAX, PID_IMAX);   // anti-windup
  return constrain((PID_KP * e + PID_KI * pidI[i] - PID_KD * spd) / 90.0f, -1.0f, 1.0f);
}

float slew(float cur, float want, float maxStep) {
  float d = want - cur;
  if (d > maxStep) d = maxStep;
  if (d < -maxStep) d = -maxStep;
  return cur + d;
}

void relockView() {                                  // ADRC: current camera attitude becomes the target
  InitialRoll = KalmanAngleRoll; InitialPitch = KalmanAnglePitch; InitialYaw = YawAngle;
  offsetDeg[0] = offsetDeg[1] = offsetDeg[2] = 0;
  ramp[0] = InitialRoll; ramp[1] = InitialPitch; ramp[2] = InitialYaw;
  resetCtl();
}
void enterPid() {                                    // PID: hold wherever the motors are now
  for (int i = 0; i < 3; i++) { cmdDeg[i] = ramp[i] = motorDeg(i); pidI[i] = 0; }
}

void controlStep(float dt) {
  float init[3] = {InitialRoll, InitialPitch, InitialYaw};
  float ang[3]  = {KalmanAngleRoll, KalmanAnglePitch, YawAngle};

  for (int i = 0; i < 3; i++) {
    float u = 0;
    if (ctrlMode == 0) {
      // ---------- ADRC: earth-referenced stabilization (IMU feedback) ----------
      ramp[i] = slew(ramp[i], init[i] + offsetDeg[i], SLEW_DPS[i] * dt);
      gTgt[i] = ramp[i]; angOut[i] = ang[i];
      if (imuFail > 5 || fabsf(gTgt[i] - ang[i]) > ERR_CUTOFF) {      // failsafe
        u = 0; uPrev[i] = 0; z1[i] = ang[i]; z2[i] = 0; z3[i] = 0;
      } else u = adrc(i, ang[i], gTgt[i], dt);
    } else {
      // ---------- PID: commanded pointing (encoder feedback) ----------
      float ma = motorDeg(i), ms = motorSpd(i);
      angOut[i] = ma;
      if (!motorValid(i)) {                                           // no feedback -> zero output, track measured
        u = 0; pidI[i] = 0; cmdDeg[i] = ramp[i] = ma;
      } else {
        ramp[i] = slew(ramp[i], constrain(cmdDeg[i], -CMD_LIMIT[i], CMD_LIMIT[i]), SLEW_DPS[i] * dt);
        u = pid(i, ma, ms, ramp[i], dt);
      }
      gTgt[i] = ramp[i];
#if !USE_CAN
      vmSpd[i] += (u * VM_ACC - VM_DAMP * vmSpd[i]) * dt;             // virtual motor for plotting
      vmPos[i] += vmSpd[i] * dt;
#endif
    }
    uCtl[i] = u;
    pwmV[i] = 1500.0f + u * 500.0f;                                   // VIRTUAL PWM (plot only)
#if USE_CAN
    if (canRun) sendAxis(i, u);                                       // REAL command
#endif
  }
}

// ======================= SERIAL COMMANDS (line based, non-blocking) =======================
void execCmd(char *s) {
  char c = s[0];
  float v = atof(s + 1);
  int ax = (c == 'R') ? 0 : (c == 'P') ? 1 : (c == 'Y') ? 2 : -1;
  if (ax >= 0) {
    if (ctrlMode == 0) offsetDeg[ax] = v; else cmdDeg[ax] = constrain(v, -CMD_LIMIT[ax], CMD_LIMIT[ax]);
    return;
  }
  switch (c) {
    case '1': ctrlMode = 0; relockView(); Serial.println("MODE: ADRC (earth stabilization) - view locked"); break;
    case '2': ctrlMode = 1; enterPid();   Serial.println("MODE: PID (commanded pointing, encoder feedback)"); break;
    case 't': if (ctrlMode == 0) { relockView(); Serial.println("view re-locked"); } break;
    case 'z': for (int i = 0; i < 3; i++) { offsetDeg[i] = 0; cmdDeg[i] = 0; } break;
    case 'd': dbg = !dbg; break;
    case 'o':
#if USE_CAN
      for (int i = 0; i < 3; i++) needZero[i] = true;
      Serial.println("encoders will zero at current position");
#endif
      break;
    case 'm':
#if USE_MAG_YAW
      for (int k = 0; k < 3; k++) { magMin[k] = 1e9f; magMax[k] = -1e9f; }
      magCalEnd = millis() + 15000;
      Serial.println("MAG CAL: rotate IMU through ALL orientations for 15 s (motors OFF)");
#else
      Serial.println("USE_MAG_YAW is 0");
#endif
      break;
    case 'e':
#if USE_CAN
      resetCtl(); motorsEnable(); Serial.println("MOTORS ENABLED");
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
void handleSerial() {
  static char buf[24]; static uint8_t n = 0;
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') { if (n) { buf[n] = 0; execCmd(buf); n = 0; } }
    else if (n < 23) buf[n++] = c;
  }
}

// ======================= SETUP / LOOP =======================
uint32_t lastImuUs, lastCtlUs, lastPrtUs, lastDbgMs;

void setup() {
  Serial.begin(115200);
  delay(500);
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(400000);

  if (!imuInit()) { Serial.println("ICM-20948 NOT FOUND (check wiring/address)"); while (1) delay(1000); }

#if USE_CAN
  twai_general_config_t g = TWAI_GENERAL_CONFIG_DEFAULT(CAN_TX_PIN, CAN_RX_PIN, TWAI_MODE_NORMAL);
  g.rx_queue_len = 20; g.tx_queue_len = 10;
  twai_timing_config_t  t = TWAI_TIMING_CONFIG_1MBITS();
  twai_filter_config_t  f = TWAI_FILTER_CONFIG_ACCEPT_ALL();
  if (twai_driver_install(&g, &t, &f) != ESP_OK || twai_start() != ESP_OK) {
    Serial.println("TWAI init FAILED"); while (1) delay(1000);
  }
#endif

  Serial.println("Keep gimbal STILL: gyro calibration...");
  float sx = 0, sy = 0, sz = 0; int n = 0;
  for (int i = 0; i < 1000; i++) {
    if (readImu()) { sx += GyroX; sy += GyroY; sz += GyroZ; n++; }
    delay(1);
  }
  if (n < 500) { Serial.println("IMU read failing during calibration"); while (1) delay(1000); }
  biasR = sx / n / GYRO_SCALE; biasP = sy / n / GYRO_SCALE; biasY = sz / n / GYRO_SCALE;

  // Warm-up, then capture 500 stationary Kalman samples as the initial world-frame target
  uint32_t last = micros();
  float sr = 0, sp = 0, sya = 0;
  for (int i = 0; i < 800; i++) {
    while (micros() - last < 1000) {}
    float dt = (micros() - last) * 1e-6f; last = micros();
    imuStep(dt);
#if USE_MAG_YAW
    if (i == 150) { float raw[3]; if (readMagRaw(raw)) { YawAngle = heading(raw); KalmanUncertaintyYaw = 1.0f; } }
#endif
    if (i >= 300) { sr += KalmanAngleRoll; sp += KalmanAnglePitch; sya += YawAngle; }
  }
  InitialRoll = sr / 500.0f; InitialPitch = sp / 500.0f; InitialYaw = sya / 500.0f;
  ramp[0] = InitialRoll; ramp[1] = InitialPitch; ramp[2] = InitialYaw;

  computeGains();
  resetCtl();
  lastImuUs = lastCtlUs = lastPrtUs = micros();
  lastDbgMs = millis();
  Serial.printf("READY (ADRC mode). InitialRoll=%.2f InitialPitch=%.2f InitialYaw=%.2f  USE_CAN=%d MAG=%d\n",
                InitialRoll, InitialPitch, InitialYaw, USE_CAN, USE_MAG_YAW);
  Serial.println("Commands: 1=ADRC 2=PID e/x=motors R/P/Y<deg> z t o m d  (end with Newline)");
}

void loop() {
  uint32_t now = micros();

  if (now - lastImuUs >= 1000) {                     // IMU + Kalman: 1000 Hz
    float dt = min((now - lastImuUs) * 1e-6f, 0.005f); lastImuUs = now;
    imuStep(dt);
  }
  if (now - lastCtlUs >= 2000) {                     // controller + output: 500 Hz
    float dt = min((now - lastCtlUs) * 1e-6f, 0.01f); lastCtlUs = now;
    controlStep(dt);
  }
  if (now - lastPrtUs >= 40000) {                    // Serial Plotter: 25 Hz
    lastPrtUs = now;
    const char *nm = (ctrlMode == 0) ? "" : "Motor";
    Serial.printf("%sRoll:%.2f,%sPitch:%.2f,%sYaw:%.2f,TargetRoll:%.2f,TargetPitch:%.2f,TargetYaw:%.2f,RollPWM:%.0f,PitchPWM:%.0f,YawPWM:%.0f\n",
                  nm, angOut[0], nm, angOut[1], nm, angOut[2],
                  gTgt[0], gTgt[1], gTgt[2], pwmV[0], pwmV[1], pwmV[2]);
  }
  handleSerial();
#if USE_CAN
  readFeedback();
#endif
#if USE_MAG_YAW
  if (dbg && millis() - lastDbgMs > 250) {           // verify mag axes: both should track when you rotate flat
    lastDbgMs = millis();
    Serial.printf("DBG heading_mag:%.1f yaw_fused:%.1f\n", lastHeading, YawAngle);
  }
#endif
}
