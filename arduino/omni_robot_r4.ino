/*
  omni_robot_r4_path.ino
  ─────────────────────────────────────────────────────────────
  Omni Robot – Arduino Uno R4 WiFi
  Path-following edition (compatible with Autobot_navigation_v3.py)

  NEW ENDPOINTS
  ─────────────
    POST /path
        Body (text/plain):
          <cm_per_pixel>;<x0>,<y0>;<x1>,<y1>;<x2>,<y2>;…
        Example:
          0.0423;640,480;600,420;540,340;480,260
        The first token is the cm/px scale.
        Each subsequent token is one waypoint pixel coordinate.
        The robot executes segments in order, then stops.

    GET  /stop   – abort path execution immediately

  EXISTING ENDPOINTS (unchanged)
  ────────────────────────────────
    GET /cmd?v=F|B|L|R|A|D|Q|E|S
    Speed prefix 1/2/3 still accepted (ignored during path follow
    – path always uses PATH_SPEED_CM_S for accurate timing).

  HOW TIMING WORKS
  ─────────────────
    distance_cm  = sqrt(dx² + dy²)  in cm  (using cm_per_pixel)
    duration_ms  = (distance_cm / PATH_SPEED_CM_S) * 1000
    The motor runs that many milliseconds then stops before the
    next segment begins.  Calibrate PATH_SPEED_CM_S by driving
    1 second at MOTOR_SPEED and measuring the distance in cm.

  DIRECTION MAPPING  (mirrors Python _vec_to_cmd)
  ──────────────────
    Screen coords: +x = right, +y = DOWN (same as OpenCV).
    dy > 0 means the next waypoint is BELOW the current one
    → robot moves FORWARD (camera is above the arena looking down).
    Adjust CAMERA_Y_INVERTED if your camera is mounted differently.

  Pin map (unchanged from omni_robot_r4.ino)
  ─────────────────
    FB: DIR=8  PWM=10  BRK=5
    LR: DIR=7  PWM=9   BRK=4
  ─────────────────────────────────────────────────────────────
*/

#include <WiFiS3.h>

// ── WiFi credentials ──────────────────────────────────────────
const char* SSID     = "MelvinJoelHotspot";
const char* PASSWORD = "GUESS_IT";

WiFiServer server(80);

// ── Motor speeds ──────────────────────────────────────────────
#define MOTOR_SPEED       140
#define MOTOR_SPEED_BACK  140

// ── Motor pins ────────────────────────────────────────────────
#define FB_DIR  8
#define FB_PWM  10
#define FB_BRK  5

#define LR_DIR  7
#define LR_PWM  9
#define LR_BRK  4

// ── Path execution config ─────────────────────────────────────
// Measure this: drive at MOTOR_SPEED for 1 second, measure cm travelled.
#define PATH_SPEED_CM_S     18.0f   // cm per second at MOTOR_SPEED

// Minimum segment length to bother driving (avoids micro-jitter).
#define MIN_SEGMENT_CM       1.5f

// Pause between segments (ms) – lets inertia settle.
#define INTER_SEGMENT_MS     80

// Maximum waypoints in one path.
#define MAX_WAYPOINTS        64

// ── Safety timeout (teleop only – suspended during path follow) ──
const unsigned long TIMEOUT_MS = 1500;

// ─────────────────────────────────────────────────────────────
//  Runtime state
// ─────────────────────────────────────────────────────────────
char          lastCommand     = 'S';
unsigned long lastCommandTime = 0;

// Path storage
struct Waypoint { int x; int y; };
Waypoint  waypoints[MAX_WAYPOINTS];
int       waypointCount  = 0;
int       currentSegment = 0;
float     cmPerPixel     = 0.04f;   // updated when path arrives

volatile bool pathActive   = false;
volatile bool pathAborted  = false;

// ─────────────────────────────────────────────────────────────
//  Motor helpers  (identical to original)
// ─────────────────────────────────────────────────────────────
void setChannel(uint8_t dirPin, uint8_t brkPin,
                uint8_t pwmPin, bool fwd, uint8_t speed) {
  analogWrite(pwmPin, 0);
  delay(30);
  digitalWrite(brkPin, LOW);
  digitalWrite(dirPin, fwd ? LOW : HIGH);
  delay(30);
  analogWrite(pwmPin, speed);
}

void setChannelInv(uint8_t dirPin, uint8_t brkPin,
                   uint8_t pwmPin, bool fwd, uint8_t speed) {
  analogWrite(pwmPin, 0);
  delay(30);
  digitalWrite(brkPin, LOW);
  digitalWrite(dirPin, fwd ? HIGH : LOW);
  delay(30);
  analogWrite(pwmPin, speed);
}

void stopChannel(uint8_t dirPin, uint8_t brkPin, uint8_t pwmPin) {
  analogWrite(pwmPin, 0);
  digitalWrite(dirPin, LOW);
  digitalWrite(brkPin, LOW);
}

// ─────────────────────────────────────────────────────────────
//  Movement functions  (identical to original)
// ─────────────────────────────────────────────────────────────
void moveForward()     { setChannelInv(FB_DIR,FB_BRK,FB_PWM,false,MOTOR_SPEED);   stopChannel(LR_DIR,LR_BRK,LR_PWM); }
void moveBackward()    { setChannelInv(FB_DIR,FB_BRK,FB_PWM,true, MOTOR_SPEED_BACK); stopChannel(LR_DIR,LR_BRK,LR_PWM); }
void moveLeft()        { stopChannel(FB_DIR,FB_BRK,FB_PWM);   setChannel(LR_DIR,LR_BRK,LR_PWM,true, MOTOR_SPEED); }
void moveRight()       { stopChannel(FB_DIR,FB_BRK,FB_PWM);   setChannel(LR_DIR,LR_BRK,LR_PWM,false,MOTOR_SPEED); }
void moveDiagFL()      { setChannelInv(FB_DIR,FB_BRK,FB_PWM,false,MOTOR_SPEED);   setChannel(LR_DIR,LR_BRK,LR_PWM,true, MOTOR_SPEED); }
void moveDiagFR()      { setChannelInv(FB_DIR,FB_BRK,FB_PWM,false,MOTOR_SPEED);   setChannel(LR_DIR,LR_BRK,LR_PWM,false,MOTOR_SPEED); }
void moveDiagBL()      { setChannelInv(FB_DIR,FB_BRK,FB_PWM,true, MOTOR_SPEED_BACK); setChannel(LR_DIR,LR_BRK,LR_PWM,true, MOTOR_SPEED); }
void moveDiagBR()      { setChannelInv(FB_DIR,FB_BRK,FB_PWM,true, MOTOR_SPEED_BACK); setChannel(LR_DIR,LR_BRK,LR_PWM,false,MOTOR_SPEED); }
void stopMotors()      { stopChannel(FB_DIR,FB_BRK,FB_PWM);   stopChannel(LR_DIR,LR_BRK,LR_PWM); }

// ─────────────────────────────────────────────────────────────
//  Direction from dx/dy  (mirrors Python _vec_to_cmd exactly)
//
//  Screen coordinates: +x = right, +y = DOWN.
//  dy > 0  → next waypoint is below robot on screen
//          → camera looks down at arena → robot moves FORWARD.
// ─────────────────────────────────────────────────────────────
char vecToCmd(int dx, int dy) {
  int ax = abs(dx), ay = abs(dy);

  if (ax > ay * 2) {
    return (dx > 0) ? 'R' : 'L';           // mostly horizontal
  }
  if (ay > ax * 2) {
    return (dy > 0) ? 'F' : 'B';           // mostly vertical
  }
  // Diagonal quadrants
  if (dx > 0 && dy < 0) return 'D';        // right + up   → FR
  if (dx < 0 && dy < 0) return 'A';        // left  + up   → FL
  if (dx > 0 && dy > 0) return 'E';        // right + down → BR
  return 'Q';                               // left  + down → BL
}

// ─────────────────────────────────────────────────────────────
//  Dispatch a command letter (same switch as original)
// ─────────────────────────────────────────────────────────────
void executeCmd(char cmd) {
  switch (cmd) {
    case 'F': moveForward();  break;
    case 'B': moveBackward(); break;
    case 'L': moveLeft();     break;
    case 'R': moveRight();    break;
    case 'A': moveDiagFL();   break;
    case 'D': moveDiagFR();   break;
    case 'Q': moveDiagBL();   break;
    case 'E': moveDiagBR();   break;
    case 'S': stopMotors();   break;
    default:
      Serial.print("[WARN] Unknown cmd: "); Serial.println(cmd);
      break;
  }
}

// ─────────────────────────────────────────────────────────────
//  Teleop command handler  (resets safety timer)
// ─────────────────────────────────────────────────────────────
void handleTeleopCmd(char cmd) {
  if (pathActive) {
    // Abort path if operator takes manual control
    pathAborted = true;
    pathActive  = false;
    stopMotors();
    Serial.println("[PATH] Aborted by manual cmd");
  }
  lastCommand     = cmd;
  lastCommandTime = millis();
  executeCmd(cmd);
  Serial.print("[CMD] "); Serial.println(cmd);
}

// ─────────────────────────────────────────────────────────────
//  PATH PARSER
//
//  Expected body format (no spaces, \n, or quotes):
//    <cm_per_pixel>;<x0>,<y0>;<x1>,<y1>;…
//
//  Example body string:
//    "0.0423;640,480;600,420;540,340"
//
//  Returns number of waypoints parsed (0 = parse error).
// ─────────────────────────────────────────────────────────────
int parsePath(const String& body) {
  waypointCount = 0;

  // ── First token: cm_per_pixel ──────────────────────────────
  int semi = body.indexOf(';');
  if (semi < 0) {
    Serial.println("[PATH] Parse error: no semicolons");
    return 0;
  }
  float scale = body.substring(0, semi).toFloat();
  if (scale <= 0.0f) {
    Serial.println("[PATH] Parse error: bad scale");
    return 0;
  }
  cmPerPixel = scale;
  Serial.print("[PATH] cm/px = "); Serial.println(cmPerPixel, 5);

  // ── Remaining tokens: x,y pairs ───────────────────────────
  int pos = semi + 1;
  while (pos < (int)body.length() && waypointCount < MAX_WAYPOINTS) {
    int nextSemi = body.indexOf(';', pos);
    String token = (nextSemi >= 0)
                   ? body.substring(pos, nextSemi)
                   : body.substring(pos);
    token.trim();
    if (token.length() == 0) { pos = nextSemi + 1; continue; }

    int comma = token.indexOf(',');
    if (comma < 0) { pos = (nextSemi >= 0) ? nextSemi + 1 : body.length(); continue; }

    int wx = token.substring(0, comma).toInt();
    int wy = token.substring(comma + 1).toInt();
    waypoints[waypointCount++] = { wx, wy };

    pos = (nextSemi >= 0) ? nextSemi + 1 : (int)body.length();
  }

  Serial.print("[PATH] Parsed "); Serial.print(waypointCount);
  Serial.println(" waypoints");
  return waypointCount;
}

// ─────────────────────────────────────────────────────────────
//  PATH EXECUTOR  (called once from loop after /path received)
//
//  Iterates through segments one at a time.
//  Each segment: compute direction + duration, drive, stop, pause.
//  Checks pathAborted flag every ~20 ms during the drive.
// ─────────────────────────────────────────────────────────────
void executePath() {
  if (waypointCount < 2) {
    Serial.println("[PATH] Not enough waypoints");
    pathActive = false;
    return;
  }

  Serial.println("[PATH] Execution started");
  pathActive  = true;
  pathAborted = false;

  for (int i = 0; i < waypointCount - 1 && !pathAborted; i++) {
    int dx = waypoints[i+1].x - waypoints[i].x;
    int dy = waypoints[i+1].y - waypoints[i].y;

    // Distance in cm
    float distPx = sqrt((float)(dx*dx) + (float)(dy*dy));
    float distCm = distPx * cmPerPixel;

    if (distCm < MIN_SEGMENT_CM) {
      Serial.print("[PATH] Seg "); Serial.print(i);
      Serial.println(" too short – skipped");
      continue;
    }

    char cmd = vecToCmd(dx, dy);
    unsigned long durationMs = (unsigned long)((distCm / PATH_SPEED_CM_S) * 1000.0f);

    Serial.print("[PATH] Seg "); Serial.print(i);
    Serial.print("  dx="); Serial.print(dx);
    Serial.print(" dy="); Serial.print(dy);
    Serial.print("  dist="); Serial.print(distCm, 1); Serial.print("cm");
    Serial.print("  cmd="); Serial.print(cmd);
    Serial.print("  dur="); Serial.print(durationMs); Serial.println("ms");

    // Drive the segment
    executeCmd(cmd);
    unsigned long t0 = millis();
    while (!pathAborted && (millis() - t0 < durationMs)) {
      // Yield 20 ms slices so the WiFi stack can breathe.
      // We do NOT call server.available() here – that would
      // require re-entrant HTTP parsing.  The /stop endpoint
      // is handled via the main loop polling below instead.
      delay(20);
    }

    stopMotors();

    if (pathAborted) break;

    // Short pause between segments
    delay(INTER_SEGMENT_MS);
  }

  stopMotors();
  pathActive = false;

  if (pathAborted) {
    Serial.println("[PATH] Aborted");
  } else {
    Serial.println("[PATH] Complete");
  }
}

// ─────────────────────────────────────────────────────────────
//  HTTP BODY READER
//  Reads Content-Length bytes after the blank line.
// ─────────────────────────────────────────────────────────────
String readBody(WiFiClient& client, const String& request) {
  // Extract Content-Length
  int clIdx = request.indexOf("Content-Length:");
  if (clIdx < 0) return "";

  int clEnd = request.indexOf("\r\n", clIdx);
  int contentLength = request.substring(clIdx + 15, clEnd).toInt();
  if (contentLength <= 0 || contentLength > 2048) return "";

  String body = "";
  unsigned long t = millis();
  while ((int)body.length() < contentLength && millis() - t < 3000) {
    if (client.available()) {
      body += (char)client.read();
    }
  }
  return body;
}

// ─────────────────────────────────────────────────────────────
//  SETUP
// ─────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n========================================");
  Serial.println("  Omni Robot – Path Following Edition");
  Serial.println("  Arduino Uno R4 WiFi");
  Serial.println("========================================");

  pinMode(FB_DIR, OUTPUT); pinMode(FB_PWM, OUTPUT); pinMode(FB_BRK, OUTPUT);
  pinMode(LR_DIR, OUTPUT); pinMode(LR_PWM, OUTPUT); pinMode(LR_BRK, OUTPUT);
  stopMotors();

  Serial.println("[INIT] Motors initialised");
  Serial.println("[PINS] FB: DIR=8 PWM=10 BRK=5");
  Serial.println("[PINS] LR: DIR=7 PWM=9  BRK=4");

  Serial.println("[WiFi] Connecting...");
  WiFi.begin(SSID, PASSWORD);
  while (WiFi.status() != WL_CONNECTED) { Serial.print("."); delay(500); }
  Serial.println("\n[WiFi] Connected!");

  IPAddress ip = WiFi.localIP();
  while (ip == IPAddress(0,0,0,0)) { delay(500); ip = WiFi.localIP(); }

  Serial.print("[WiFi] ROBOT IP: "); Serial.println(ip);
  server.begin();
  Serial.println("[HTTP] Server started on port 80");
  Serial.println("[READY] Endpoints: /cmd  /path  /stop");

  lastCommandTime = millis();
}

// ─────────────────────────────────────────────────────────────
//  LOOP
// ─────────────────────────────────────────────────────────────
void loop() {
  WiFiClient client = server.available();

  if (client) {
    String request = "";
    unsigned long t = millis();
    while (client.connected() && millis() - t < 3000) {
      if (client.available()) {
        char c = client.read();
        request += c;
        if (request.endsWith("\r\n\r\n")) break;
      }
    }

    // ── Route request ──────────────────────────────────────

    // GET /cmd?v=X  – teleop (same as original)
    if (request.indexOf("GET /cmd") >= 0) {
      char cmd = 'S';
      int idx = request.indexOf("v=");
      if (idx >= 0) cmd = request.charAt(idx + 2);
      // Strip speed prefix (1F → F, 2B → B, etc.)
      if (cmd >= '1' && cmd <= '3') {
        idx = request.indexOf("v=") + 3;
        if (idx < (int)request.length()) cmd = request.charAt(idx);
      }
      handleTeleopCmd(cmd);
      client.println("HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nConnection: close\r\n\r\nOK");
      client.stop();

    // GET /stop  – abort path
    } else if (request.indexOf("GET /stop") >= 0) {
      pathAborted = true;
      pathActive  = false;
      stopMotors();
      Serial.println("[STOP] Manual abort via /stop");
      client.println("HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nConnection: close\r\n\r\nSTOPPED");
      client.stop();

    // POST /path  – receive and execute path
    } else if (request.indexOf("POST /path") >= 0) {
      String body = readBody(client, request);
      client.println("HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nConnection: close\r\n\r\nACCEPTED");
      client.stop();   // respond BEFORE blocking on path execution

      if (body.length() > 0) {
        int n = parsePath(body);
        if (n >= 2) {
          executePath();   // blocking – runs to completion or abort
        } else {
          Serial.println("[PATH] Parse failed or too few waypoints");
        }
      }

    // Unknown request
    } else {
      client.println("HTTP/1.1 404 Not Found\r\nConnection: close\r\n\r\nNot Found");
      client.stop();
    }
  }

  // Safety auto-stop (teleop only; skip if path is active)
  if (!pathActive
      && millis() - lastCommandTime > TIMEOUT_MS
      && lastCommand != 'S') {
    Serial.println("[TIMEOUT] Auto STOP");
    handleTeleopCmd('S');
  }

  delay(10);
}