#include <Servo.h>

Servo servo;
const int servoPin = 9;
String inputString = "";
bool stringComplete = false;

void setup() {
  // Start serial first
  Serial.begin(115200);
  
  // Wait for serial port to connect
  while (!Serial) {
    ; // Wait for serial port to connect. Needed for native USB
  }
  
  // Clear any garbage data
  while(Serial.available()) {
    Serial.read();
  }
  
  // Setup servo
  servo.attach(servoPin);
  servo.write(90);  // Initialize at neutral position
  delay(1000);  // Give servo time to reach position
  
  // Send ready signal multiple times to ensure it's received
  for(int i = 0; i < 3; i++) {
    Serial.println("READY");
    delay(100);
  }
  
  inputString.reserve(200);
}

void loop() {
  if (stringComplete) {
    // Convert the incoming string to an integer
    int angle = inputString.toInt();
    
    // Clamp the angle to valid range
    angle = constrain(angle, 0, 180);
    
    // Move servo
    servo.write(angle);
    
    // Send confirmation
    Serial.print("Moved to angle: ");
    Serial.println(angle);
    
    // Clear for next command
    inputString = "";
    stringComplete = false;
  }
  
  // Keep the communication alive
  if (Serial.available() == 0) {
    delay(10);  // Small delay when no data
  }
}

void serialEvent() {
  while (Serial.available()) {
    char inChar = (char)Serial.read();
    
    // If newline, set flag
    if (inChar == '\n') {
      stringComplete = true;
    } else {
      // Add character to string
      inputString += inChar;
    }
  }
} 