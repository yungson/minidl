import sys
import numpy as np
import pytest 
sys.path.insert(0,"../")
import minidl as mdl


def test_crossentropy():
    lossfn = CrossEntropyLoss()