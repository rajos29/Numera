#include <WiFi.h>

const unsigned long SERIAL_BAUD = 115200;
const unsigned long CONNECT_TIMEOUT_MS = 20000;

struct NetworkInfo {
  String ssid;
  int32_t rssi;
  wifi_auth_mode_t encryption;
};

const int MAX_NETWORKS = 40;
NetworkInfo networks[MAX_NETWORKS];
int networkCount = 0;

String readLine(const char *prompt) {
  Serial.print(prompt);

  while (!Serial.available()) {
    delay(10);
  }

  String line = Serial.readStringUntil('\n');
  line.trim();
  return line;
}

String encryptionName(wifi_auth_mode_t type) {
  switch (type) {
    case WIFI_AUTH_OPEN:
      return "open";
    case WIFI_AUTH_WEP:
      return "WEP";
    case WIFI_AUTH_WPA_PSK:
      return "WPA";
    case WIFI_AUTH_WPA2_PSK:
      return "WPA2";
    case WIFI_AUTH_WPA_WPA2_PSK:
      return "WPA/WPA2";
    case WIFI_AUTH_WPA2_ENTERPRISE:
      return "WPA2-Enterprise";
    case WIFI_AUTH_WPA3_PSK:
      return "WPA3";
    case WIFI_AUTH_WPA2_WPA3_PSK:
      return "WPA2/WPA3";
    default:
      return "unknown";
  }
}

void printHelp() {
  Serial.println();
  Serial.println("Commands:");
  Serial.println("  r       rescan Wi-Fi networks");
  Serial.println("  1..N    connect to a listed network");
  Serial.println("  ?       show this help");
  Serial.println();
}

void scanNetworks() {
  Serial.println();
  Serial.println("Scanning for Wi-Fi networks...");

  WiFi.disconnect(true);
  delay(250);

  int found = WiFi.scanNetworks(false, true);
  networkCount = 0;

  if (found <= 0) {
    Serial.println("No networks found. Type r to try again.");
    return;
  }

  int limit = min(found, MAX_NETWORKS);
  for (int i = 0; i < limit; i++) {
    networks[i].ssid = WiFi.SSID(i);
    networks[i].rssi = WiFi.RSSI(i);
    networks[i].encryption = WiFi.encryptionType(i);
    networkCount++;
  }

  Serial.println();
  Serial.println("#   RSSI   Security       SSID");
  Serial.println("-----------------------------------------------");

  for (int i = 0; i < networkCount; i++) {
    Serial.printf("%-3d %-6ld %-14s %s\n",
                  i + 1,
                  networks[i].rssi,
                  encryptionName(networks[i].encryption).c_str(),
                  networks[i].ssid.length() ? networks[i].ssid.c_str() : "<hidden>");
  }

  if (found > MAX_NETWORKS) {
    Serial.printf("\nShowing strongest %d of %d networks.\n", MAX_NETWORKS, found);
  }

  WiFi.scanDelete();
}

void printConnectionResult(wl_status_t status) {
  Serial.println();

  switch (status) {
    case WL_CONNECTED:
      Serial.println("Connected.");
      Serial.print("SSID: ");
      Serial.println(WiFi.SSID());
      Serial.print("IP address: ");
      Serial.println(WiFi.localIP());
      Serial.print("Signal: ");
      Serial.print(WiFi.RSSI());
      Serial.println(" dBm");
      break;
    case WL_NO_SSID_AVAIL:
      Serial.println("Connection failed: SSID not available.");
      break;
    case WL_CONNECT_FAILED:
      Serial.println("Connection failed: authentication or association failed.");
      break;
    case WL_CONNECTION_LOST:
      Serial.println("Connection failed: connection was lost.");
      break;
    case WL_DISCONNECTED:
      Serial.println("Connection failed: timed out or disconnected.");
      break;
    default:
      Serial.printf("Connection failed: WiFi status %d.\n", status);
      break;
  }
}

void connectToNetwork(int index) {
  if (index < 0 || index >= networkCount) {
    Serial.println("That network number is not in the list.");
    return;
  }

  String ssid = networks[index].ssid;
  wifi_auth_mode_t encryption = networks[index].encryption;

  if (!ssid.length()) {
    ssid = readLine("Hidden SSID selected. Enter SSID: ");
    if (!ssid.length()) {
      Serial.println("SSID cannot be empty.");
      return;
    }
  }

  String password;
  if (encryption != WIFI_AUTH_OPEN) {
    password = readLine("Enter password: ");
  }

  Serial.println();
  Serial.print("Connecting to ");
  Serial.print(ssid);
  Serial.print(" ");

  WiFi.disconnect(true);
  delay(250);
  WiFi.mode(WIFI_STA);

  if (encryption == WIFI_AUTH_OPEN && password.length() == 0) {
    WiFi.begin(ssid.c_str());
  } else {
    WiFi.begin(ssid.c_str(), password.c_str());
  }

  unsigned long startedAt = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - startedAt < CONNECT_TIMEOUT_MS) {
    Serial.print(".");
    delay(500);
  }

  printConnectionResult(WiFi.status());
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  Serial.setTimeout(120000);
  delay(1000);

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);

  Serial.println();
  Serial.println("ESP32 Wi-Fi terminal selector");
  Serial.println("Open Serial Monitor at 115200 baud with Newline enabled.");

  printHelp();
  scanNetworks();
}

void loop() {
  String command = readLine("\nSelect network number, r to rescan, or ? for help: ");

  if (command.equalsIgnoreCase("r")) {
    scanNetworks();
    return;
  }

  if (command == "?") {
    printHelp();
    return;
  }

  int selection = command.toInt();
  if (selection <= 0) {
    Serial.println("Enter a listed network number, r, or ?.");
    return;
  }

  connectToNetwork(selection - 1);
}
