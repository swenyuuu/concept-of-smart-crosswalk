int trigPin = 8;          // 超音波感測器 Trig 腳位
int echoPin = 9;          // 超音波感測器 Echo 腳位
const int stopPin = 7;    // 車道紅燈
const int redPin = 11;    // 行人紅燈
const int greenPin = 12;  // 行人綠燈
long duration, cm;
int stat, nstable=6, fstable=6;
bool ntimes=true, ftimes=true;
String data;
unsigned long beginTime = millis();

void setup() {
  Serial.begin (115200);
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);
  pinMode(12, OUTPUT);
  pinMode(11, OUTPUT);
  pinMode(7, OUTPUT);
  digitalWrite(redPin, HIGH);
  digitalWrite(stopPin, HIGH);
}

// 計算距離
void measureDistance() {
    digitalWrite(trigPin, LOW);
    delayMicroseconds(5);
    digitalWrite(trigPin, HIGH);
    delayMicroseconds(10);
    digitalWrite(trigPin, LOW);
    duration = pulseIn(echoPin, HIGH);
    cm = (duration / 2) / 29.1;
}

// 控制燈號
void controlLED(){
  data = Serial.readStringUntil('\r');
  if (data=="g"){
    digitalWrite(stopPin, HIGH);
    delay(250);
    digitalWrite(redPin, LOW);
    digitalWrite(greenPin, HIGH);
  }
  if (data=="r"){
    digitalWrite(redPin, HIGH);
    digitalWrite(greenPin, LOW);
    delay(250);
    digitalWrite(stopPin, LOW);
    beginTime=millis();
  }
}

void loop() {
  measureDistance();  // 計算距離
    if (cm > 0 && cm < 100) {
      if (fstable>0){
        fstable=fstable-1;
        stat=0;
      }else{
        stat=1;
        nstable=6;
        if (ntimes==true){
          Serial.println(stat);
          ntimes=false;
          ftimes=true;
        }
      }
    }else{
      if (nstable>0){
        nstable=nstable-1;
        stat=1;
      }else{
        stat=0;
        fstable=6;
        if (ftimes==true){
          Serial.println(stat);
          ftimes=false;
          ntimes=true;
        }
      }
    }
  if (Serial.available()>0){controlLED();}  // 控制燈號
  if (data!="g" && millis()-beginTime>=1000){   //控制車道燈閃爍
    digitalWrite(stopPin, !digitalRead(stopPin));
    beginTime=millis();
  }
  delay(100);
}
