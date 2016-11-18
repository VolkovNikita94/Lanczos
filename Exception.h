#pragma once
// Exception class used for argument check.

// Basic inclusions.
#include <exception>
#include <string>

class invalid_args : public std::exception {
public:
	// Constructor.
	invalid_args();

	// Initialization.
	void init();

	// Get string.
	std::string get();
private:
	std::string message;
};

// Function implementations.
// -------------------------

// Constuctor
invalid_args::invalid_args() {
}

// Initialization.
void invalid_args::init() {
	message = "Invalid argument relations. Check arguments thoroughly !\n";
}

// Get string.
std::string invalid_args::get() {
	return message;
}