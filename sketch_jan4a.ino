#include <Arduino.h>
#include <esp_camera.h>
#include "esp_system.h"

// TensorFlow Lite Micro includes
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Include our converted model
#include "model_esp32.h"

// Pin definitions for ESP32-CAM
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22
#define FLASH_GPIO_NUM     4

// Constants - reduced sizes for memory optimization
constexpr int kTensorArenaSize = 100 * 1024;  // Reduced to 100K
constexpr int kNumChannels = 3;
constexpr int kImageWidth = 96;
constexpr int kImageHeight = 96;
constexpr float kThreshold = 0.5f;

// Global variables
struct {
    uint8_t *tensor_arena = nullptr;
    const tflite::Model* model = nullptr;
    tflite::MicroInterpreter* interpreter = nullptr;
    TfLiteTensor* input_tensor = nullptr;
    TfLiteTensor* output_tensor = nullptr;
} tf_data;

void printMemory() {
    Serial.printf("Total heap: %d\n", ESP.getHeapSize());
    Serial.printf("Free heap: %d\n", ESP.getFreeHeap());
}

bool initCamera() {
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = Y2_GPIO_NUM;
    config.pin_d1 = Y3_GPIO_NUM;
    config.pin_d2 = Y4_GPIO_NUM;
    config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM;
    config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM;
    config.pin_d7 = Y9_GPIO_NUM;
    config.pin_xclk = XCLK_GPIO_NUM;
    config.pin_pclk = PCLK_GPIO_NUM;
    config.pin_vsync = VSYNC_GPIO_NUM;
    config.pin_href = HREF_GPIO_NUM;
    config.pin_sscb_sda = SIOD_GPIO_NUM;
    config.pin_sscb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn = PWDN_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_RGB565;
    config.frame_size = FRAMESIZE_96X96;
    config.jpeg_quality = 12;
    config.fb_count = 1;

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("Camera init failed with error 0x%x\n", err);
        return false;
    }
    return true;
}

bool initTensorFlow() {
    // Try to allocate memory using different strategies
    tf_data.tensor_arena = (uint8_t*)malloc(kTensorArenaSize);
    
    if (!tf_data.tensor_arena) {
        // If first attempt fails, try heap_caps_malloc
        tf_data.tensor_arena = (uint8_t*)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
    }
    
    if (!tf_data.tensor_arena) {
        Serial.println("Failed to allocate tensor arena");
        return false;
    }

    Serial.println("Tensor arena allocated successfully");

    tf_data.model = tflite::GetModel(model_esp32_tflite);
    if (tf_data.model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Model version mismatch!");
        return false;
    }

    static tflite::MicroMutableOpResolver<5> micro_op_resolver;
    micro_op_resolver.AddConv2D();
    micro_op_resolver.AddMaxPool2D();
    micro_op_resolver.AddReshape();
    micro_op_resolver.AddFullyConnected();
    micro_op_resolver.AddSoftmax();

    static tflite::MicroInterpreter static_interpreter(
        tf_data.model, micro_op_resolver, tf_data.tensor_arena, 
        kTensorArenaSize);
    tf_data.interpreter = &static_interpreter;

    if (tf_data.interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("AllocateTensors() failed");
        return false;
    }

    tf_data.input_tensor = tf_data.interpreter->input(0);
    tf_data.output_tensor = tf_data.interpreter->output(0);

    Serial.println("TensorFlow initialized successfully");
    return true;
}

bool captureAndProcessImage() {
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
        Serial.println("Camera capture failed");
        return false;
    }

    uint16_t* rgb565_buffer = (uint16_t*)fb->buf;
    float* input_buffer = tf_data.input_tensor->data.f;

    for (int y = 0; y < kImageHeight; y++) {
        for (int x = 0; x < kImageWidth; x++) {
            uint16_t pixel = rgb565_buffer[y * kImageWidth + x];
            
            uint8_t r = ((pixel >> 11) & 0x1F) << 3;
            uint8_t g = ((pixel >> 5) & 0x3F) << 2;
            uint8_t b = (pixel & 0x1F) << 3;

            int idx = (y * kImageWidth + x) * kNumChannels;
            input_buffer[idx + 0] = r / 255.0f;
            input_buffer[idx + 1] = g / 255.0f;
            input_buffer[idx + 2] = b / 255.0f;
        }
    }

    esp_camera_fb_return(fb);

    if (tf_data.interpreter->Invoke() != kTfLiteOk) {
        Serial.println("Invoke failed!");
        return false;
    }

    float confidence = tf_data.output_tensor->data.f[0];
    Serial.printf("Confidence: %.3f\n", confidence);
    Serial.printf("Detection: %s\n", confidence > kThreshold ? "Caries Detected!" : "No Caries");

    return true;
}

void controlFlash(bool on) {
    digitalWrite(FLASH_GPIO_NUM, on ? HIGH : LOW);
}

void setup() {
    Serial.begin(115200);
    delay(1000); // Give serial time to initialize
    Serial.println("\nStarting Dental Caries Detection");

    printMemory();  // Print initial memory state

    pinMode(FLASH_GPIO_NUM, OUTPUT);
    digitalWrite(FLASH_GPIO_NUM, LOW);

    if (!initCamera()) {
        Serial.println("Failed to initialize camera!");
        return;
    }

    Serial.println("Camera initialized successfully");
    printMemory();  // Print memory after camera init

    if (!initTensorFlow()) {
        Serial.println("Failed to initialize TensorFlow!");
        return;
    }

    printMemory();  // Print final memory state
    Serial.println("Setup completed successfully!");
}

void loop() {
    if (Serial.available()) {
        char cmd = Serial.read();
        bool success;
        
        switch (cmd) {
            case 'd': // Detect
                Serial.println("Starting detection...");
                controlFlash(true);
                delay(100);
                success = captureAndProcessImage();
                controlFlash(false);
                if (success) {
                    Serial.println("Detection completed");
                } else {
                    Serial.println("Detection failed");
                }
                printMemory();  // Print memory after detection
                break;

            case 'f': { // Toggle flash
                static bool flash_state = false;
                flash_state = !flash_state;
                controlFlash(flash_state);
                Serial.println(flash_state ? "Flash ON" : "Flash OFF");
                break;
            }

            case 'm': // Print memory stats
                printMemory();
                break;
        }
    }
}