#!/usr/bin/env python3
"""Utility functions for LLM Reasoning Distribution analysis"""

# Defining colours
colour_dict = {
    "GREEN":     '\033[0;32m',
    "YELLOW":    '\033[1;33m',
    "RED":       '\033[0;31m',
    "BLUE":      '\033[0;34m',
    "NC":        '\033[0m'
}

def colour_print(text, colour):
    """colour print to terminal"""
    if not hasattr(colour, "upper"):
        raise AttributeError(f"Error: Colour {colour} passed does not have .upper() as an attribute. Is this supposed to be a string?") 
    
    if colour.upper() not in colour_dict.keys():
        raise ValueError(f"Error: Colour {colour} passed is not in the available set of colours: {', '.join(colour_dict.keys())}")
    
    colour_code = colour_dict[colour.upper()]
    nc_code = colour_dict["NC"]
    print(f"{colour_code}{text}{nc_code}")

def print_error(text):
    colour_print(text, "RED")
