#ifndef ARDUINO_STUBS_H
#define ARDUINO_STUBS_H

// This is a helper file for code editors/linters only
// It's not used by the actual Arduino compiler

#include <stdint.h>
#include <stddef.h>

#ifndef NULL
#define NULL 0
#endif

// Arduino's Serial object
class SerialClass {
public:
    void begin(long baud) {}
    int available() { return 0; }
    char read() { return 0; }
    void print(const char* str) {}
    void print(int num) {}
    void print(unsigned long num) {}
    void print(float num) {}
    void println(const char* str) {}
    void println(int num) {}
    void println(unsigned long num) {}
    void println() {}
    void flush() {}
};

// Declare the global Serial object
extern SerialClass Serial;

// Time functions
unsigned long millis();

// String functions
int strcmp(const char* s1, const char* s2);
int strncmp(const char* s1, const char* s2, size_t n);
char* strtok(char* str, const char* delim);
char* strchr(const char* str, int c);
int atoi(const char* str);

#endif // ARDUINO_STUBS_H 