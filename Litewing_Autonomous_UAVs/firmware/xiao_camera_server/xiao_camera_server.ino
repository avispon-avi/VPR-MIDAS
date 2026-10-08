/*
 * XIAO ESP32-S3 Sense — HTTP camera server with robust WiFi
 *
 * Features:
 *   - Scans nearby WiFi networks on boot (diagnostic)
 *   - Tries multiple SSIDs with timeout (no infinite hang)
 *   - Falls back to AP mode if nothing connects (XIAO-CAM / 12345678)
 *   - Prints detailed WiFi status codes for debugging
 *
 * Endpoints (IP is printed on Serial Monitor @115200):
 *   http://<IP>/          -> status page
 *   http://<IP>/capture   -> one JPEG frame
 *   http://<IP>/status    -> plain-text "ok"
 *   http://<IP>/scan      -> rescan nearby WiFi networks (diagnostic)
 *
 * IMPORTANT: XIAO ESP32-S3 only supports 2.4 GHz WiFi!
 *   Phone hotspots often default to 5 GHz — you MUST force 2.4 GHz:
 *     Android: Settings > Hotspot > Band > 2.4 GHz
 *     iPhone:  Settings > Personal Hotspot > Maximize Compatibility = ON
 *     Windows: Settings > Mobile hotspot > Network band > 2.4 GHz
 *
 * Arduino IDE setup (critical settings):
 *   1. Boards Manager: install "esp32 by Espressif Systems" (2.0.14+)
 *   2. Tools -> Board  : XIAO_ESP32S3
 *   3. Tools -> PSRAM  : OPI PSRAM        <-- camera WILL fail without this
 *   4. Tools -> USB CDC On Boot: Enabled  <-- so Serial prints appear over USB
 */

#include "esp_camera.h"
#include <WiFi.h>
#include <WebServer.h>
#include "esp_wifi.h"   // for esp_wifi_set_ps()

// ============== WiFi CREDENTIALS ==============
// Add all your networks here. The board tries each one in order.
// IMPORTANT: Hotspot MUST be set to 2.4 GHz band!
struct WiFiCred {
  const char *ssid;
  const char *pass;
};

WiFiCred wifiList[] = {
  {"YOUR_HOTSPOT_NAME", "YOUR_PASSWORD"},     // ← EDIT THIS: your WiFi/hotspot name & password
  // {"SecondNetwork", "password2"},           // ← Optional: add backup networks
  // {"ThirdNetwork", "password3"},
};
const int WIFI_COUNT = sizeof(wifiList) / sizeof(wifiList[0]);

// AP fallback — if NO WiFi connects, the XIAO creates its own network
const char *AP_SSID = "XIAO-CAM";
const char *AP_PASS = "12345678";   // min 8 chars for WPA2

// WiFi connection settings
#define WIFI_TIMEOUT_SEC    20      // seconds to wait per SSID
#define WIFI_MAX_RETRIES    3       // retry cycles through the entire list
// ==============================================

#define FRAME_SIZE FRAMESIZE_QVGA       // 320x240 for max stability
#define JPEG_QUALITY 15                 // 0-63, lower = better quality

// XIAO ESP32-S3 Sense (OV2640) camera pin map
#define PWDN_GPIO_NUM  -1
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM  10
#define SIOD_GPIO_NUM  40
#define SIOC_GPIO_NUM  39
#define Y9_GPIO_NUM    48
#define Y8_GPIO_NUM    11
#define Y7_GPIO_NUM    12
#define Y6_GPIO_NUM    14
#define Y5_GPIO_NUM    16
#define Y4_GPIO_NUM    18
#define Y3_GPIO_NUM    17
#define Y2_GPIO_NUM    15
#define VSYNC_GPIO_NUM 38
#define HREF_GPIO_NUM  47
#define PCLK_GPIO_NUM  13

WebServer server(80);
bool apMode = false;   // true if we fell back to AP mode

// ---- Helpers ----

const char* wifiStatusStr(wl_status_t s) {
  switch (s) {
    case WL_IDLE_STATUS:     return "IDLE";
    case WL_NO_SSID_AVAIL:   return "NO_SSID_AVAIL (network not found - check 2.4GHz!)";
    case WL_SCAN_COMPLETED:  return "SCAN_COMPLETED";
    case WL_CONNECTED:       return "CONNECTED";
    case WL_CONNECT_FAILED:  return "CONNECT_FAILED (wrong password?)";
    case WL_CONNECTION_LOST: return "CONNECTION_LOST";
    case WL_DISCONNECTED:    return "DISCONNECTED";
    default:                 return "UNKNOWN";
  }
}

void scanNetworks() {
  Serial.println("\n===== SCANNING NEARBY WiFi NETWORKS =====");
  Serial.println("(XIAO can ONLY see 2.4 GHz networks!)");
  int n = WiFi.scanNetworks();
  if (n == 0) {
    Serial.println("  No networks found! Is the antenna connected?");
  } else {
    Serial.printf("  Found %d networks:\n", n);
    for (int i = 0; i < n; i++) {
      Serial.printf("  [%2d] %-32s  Ch:%2d  RSSI:%4d dBm  %s\n",
                     i + 1,
                     WiFi.SSID(i).c_str(),
                     WiFi.channel(i),
                     WiFi.RSSI(i),
                     (WiFi.encryptionType(i) == WIFI_AUTH_OPEN) ? "Open" : "Encrypted");
    }
    // Check if any of our configured SSIDs appear
    Serial.println("\n  Matching against configured SSIDs:");
    for (int w = 0; w < WIFI_COUNT; w++) {
      bool found = false;
      for (int i = 0; i < n; i++) {
        if (WiFi.SSID(i) == wifiList[w].ssid) {
          found = true;
          Serial.printf("    \"%s\" -> FOUND (RSSI: %d dBm)\n",
                        wifiList[w].ssid, WiFi.RSSI(i));
          break;
        }
      }
      if (!found) {
        Serial.printf("    \"%s\" -> NOT VISIBLE (off? 5GHz? out of range?)\n",
                      wifiList[w].ssid);
      }
    }
  }
  WiFi.scanDelete();
  Serial.println("==========================================\n");
}

bool tryConnect(const char *ssid, const char *pass) {
  Serial.printf("[WiFi] Trying: \"%s\" ... ", ssid);
  WiFi.disconnect(true);
  delay(100);
  WiFi.begin(ssid, pass);

  int elapsed = 0;
  while (WiFi.status() != WL_CONNECTED && elapsed < WIFI_TIMEOUT_SEC * 2) {
    delay(500);
    elapsed++;
    if (elapsed % 4 == 0) {
      // Print status every 2 seconds
      Serial.printf("\n  [%ds] status: %s", elapsed / 2, wifiStatusStr(WiFi.status()));
    } else {
      Serial.print(".");
    }
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("\n[WiFi] CONNECTED! IP: %s\n", WiFi.localIP().toString().c_str());
    esp_wifi_set_ps(WIFI_PS_NONE);  // Disable power-save — keeps radio always-on
    Serial.println("[WiFi] Power-save DISABLED (radio always-on for low latency)");
    return true;
  } else {
    Serial.printf("\n[WiFi] FAILED after %ds — status: %s\n",
                  WIFI_TIMEOUT_SEC, wifiStatusStr(WiFi.status()));
    WiFi.disconnect(true);
    return false;
  }
}

void startAPFallback() {
  Serial.println("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
  Serial.println("!! ALL WiFi connections failed!");
  Serial.println("!! Starting AP (Access Point) fallback mode.");
  Serial.printf( "!! Connect your PC to WiFi: \"%s\"\n", AP_SSID);
  Serial.printf( "!! Password: %s\n", AP_PASS);
  Serial.println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");

  WiFi.mode(WIFI_AP);
  WiFi.softAP(AP_SSID, AP_PASS);
  delay(500);
  apMode = true;

  Serial.printf("[AP] IP address: %s\n", WiFi.softAPIP().toString().c_str());
  Serial.printf("[AP] capture endpoint: http://%s/capture\n",
                WiFi.softAPIP().toString().c_str());
  Serial.println("[AP] Connect your PC to the XIAO-CAM WiFi, then open the URL above.");
  Serial.println("\nCommon fixes to use STA mode instead:");
  Serial.println("  1. Phone hotspot: force 2.4 GHz band (most common issue!)");
  Serial.println("  2. Windows hotspot: Settings > Mobile hotspot > Band > 2.4 GHz");
  Serial.println("  3. Check SSID spelling and password");
  Serial.println("  4. Move XIAO closer to the hotspot");
}

// ---- Camera ----

bool initCamera() {
  camera_config_t cfg = {};
  cfg.ledc_channel = LEDC_CHANNEL_0;
  cfg.ledc_timer   = LEDC_TIMER_0;
  cfg.pin_d0 = Y2_GPIO_NUM;  cfg.pin_d1 = Y3_GPIO_NUM;
  cfg.pin_d2 = Y4_GPIO_NUM;  cfg.pin_d3 = Y5_GPIO_NUM;
  cfg.pin_d4 = Y6_GPIO_NUM;  cfg.pin_d5 = Y7_GPIO_NUM;
  cfg.pin_d6 = Y8_GPIO_NUM;  cfg.pin_d7 = Y9_GPIO_NUM;
  cfg.pin_xclk = XCLK_GPIO_NUM;
  cfg.pin_pclk = PCLK_GPIO_NUM;
  cfg.pin_vsync = VSYNC_GPIO_NUM;
  cfg.pin_href = HREF_GPIO_NUM;
  cfg.pin_sccb_sda = SIOD_GPIO_NUM;
  cfg.pin_sccb_scl = SIOC_GPIO_NUM;
  cfg.pin_pwdn = PWDN_GPIO_NUM;
  cfg.pin_reset = RESET_GPIO_NUM;
  cfg.xclk_freq_hz = 20000000;
  cfg.pixel_format = PIXFORMAT_JPEG;
  cfg.frame_size = FRAME_SIZE;
  cfg.jpeg_quality = JPEG_QUALITY;
  cfg.fb_count = 2;
  cfg.fb_location = CAMERA_FB_IN_PSRAM;
  cfg.grab_mode = CAMERA_GRAB_LATEST;

  return esp_camera_init(&cfg) == ESP_OK;
}

void handleCapture() {
  camera_fb_t *fb = esp_camera_fb_get();

  // Retry up to 3 times if capture fails
  for (int i = 0; i < 3 && !fb; i++) {
    delay(100);
    fb = esp_camera_fb_get();
  }

  if (!fb) {
    server.send(503, "text/plain", "capture failed");
    return;
  }
  server.send_P(200, "image/jpeg", (const char *)fb->buf, fb->len);
  esp_camera_fb_return(fb);
}

void handleScan() {
  String html = "<h3>WiFi Scan Results</h3><table border='1'><tr><th>#</th><th>SSID</th><th>Ch</th><th>RSSI</th><th>Auth</th></tr>";
  int n = WiFi.scanNetworks();
  for (int i = 0; i < n; i++) {
    html += "<tr><td>" + String(i+1) + "</td><td>" + WiFi.SSID(i) + "</td><td>"
          + String(WiFi.channel(i)) + "</td><td>" + String(WiFi.RSSI(i))
          + " dBm</td><td>" + ((WiFi.encryptionType(i) == WIFI_AUTH_OPEN) ? "Open" : "Encrypted")
          + "</td></tr>";
  }
  html += "</table><p>XIAO only sees 2.4 GHz networks. If your hotspot isn't listed, switch it to 2.4 GHz.</p>";
  WiFi.scanDelete();
  server.send(200, "text/html", html);
}

// ---- Setup ----

void setup() {
  Serial.begin(115200);
  delay(2000);  // Extra delay for USB CDC to initialize

  Serial.println("\n\n========================================");
  Serial.println("  XIAO ESP32-S3 Sense Camera Server");
  Serial.println("========================================\n");

  // PSRAM check
  if (!psramFound()) {
    Serial.println("FATAL: PSRAM not found. Set Tools > PSRAM > OPI PSRAM and re-flash.");
    while (true) delay(1000);
  }
  Serial.printf("PSRAM: %d bytes free\n", ESP.getFreePsram());

  // Camera init
  if (!initCamera()) {
    Serial.println("FATAL: camera init failed. Check the expansion board is seated.");
    while (true) delay(1000);
  }
  Serial.println("Camera: OK");

  // Fix the 180-degree physical rotation at the hardware level
  sensor_t * s = esp_camera_sensor_get();
  if (s != NULL) {
    s->set_vflip(s, 1);    // Flip vertically
    s->set_hmirror(s, 0);  // Disable horizontal mirroring to fix the mirror effect
  }

  // Warm up the camera and clear initial bad frames
  for (int i = 0; i < 3; i++) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (fb) esp_camera_fb_return(fb);
    delay(50);
  }

  // ---- WiFi ----
  WiFi.mode(WIFI_STA);

  // Step 1: Scan to see what's actually visible
  scanNetworks();

  // Step 2: Try each SSID with retries
  bool connected = false;
  for (int retry = 0; retry < WIFI_MAX_RETRIES && !connected; retry++) {
    if (retry > 0) {
      Serial.printf("\n--- Retry cycle %d/%d ---\n", retry + 1, WIFI_MAX_RETRIES);
    }
    for (int w = 0; w < WIFI_COUNT && !connected; w++) {
      connected = tryConnect(wifiList[w].ssid, wifiList[w].pass);
    }
  }

  // Step 3: If nothing worked, fall back to AP mode
  if (!connected) {
    startAPFallback();
  }

  // ---- HTTP server ----
  String ip = apMode ? WiFi.softAPIP().toString() : WiFi.localIP().toString();

  server.on("/", HTTP_GET, [ip]() {
    String mode = apMode ? " (AP MODE — connect PC to XIAO-CAM WiFi)" : " (STA mode)";
    String html = "<h3>XIAO Sense Camera Server" + mode + "</h3>"
                  "<p><a href='/capture'>/capture</a> - one JPEG frame</p>"
                  "<p><a href='/status'>/status</a> - plain text ok</p>"
                  "<p><a href='/scan'>/scan</a> - rescan WiFi networks</p>"
                  "<p>IP: " + ip + "</p>";
    server.send(200, "text/html", html);
  });
  server.on("/capture", HTTP_GET, handleCapture);
  server.on("/status",  HTTP_GET, []() { server.send(200, "text/plain", "ok"); });
  server.on("/scan",    HTTP_GET, handleScan);
  server.begin();

  Serial.println("\n========================================");
  Serial.printf("  HTTP server started on %s\n", ip.c_str());
  Serial.printf("  capture: http://%s/capture\n", ip.c_str());
  Serial.printf("  scan:    http://%s/scan\n", ip.c_str());
  Serial.println("========================================\n");
}

void loop() {
  server.handleClient();

  // Periodically print status in AP mode so user knows it's alive
  static unsigned long lastPrint = 0;
  if (apMode && millis() - lastPrint > 10000) {
    lastPrint = millis();
    Serial.printf("[AP] Waiting for connection... clients: %d | IP: %s\n",
                  WiFi.softAPgetStationNum(),
                  WiFi.softAPIP().toString().c_str());
  }
}
