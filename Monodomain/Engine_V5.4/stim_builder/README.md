# StimBuilder

Generates stimulation protocol maps from images for **engine_V5.4**.

## Overview

Converts image-based designs into spatial stimulation patterns. Each distinct color in the input image represents a unique stimulation region.

## Workflow

1. **Identify**: Auto-detect all distinct colors in the image
2. **Label**: User assigns names to each stimulation region
3. **Specify**: User defines stimulation parameters per region

## Stimulation Parameters

| Parameter | Options | Description |
|-----------|---------|-------------|
| **Type** | Voltage clamp / Current injection | Stimulation mode |
| **Duration** | (ms) | How long the stimulus is applied |
| **Amplitude** | (mV or μA) | Strength of stimulus |
| **Target** | Intracellular / Extracellular | *Future support* |
| **Name** | User-defined | Label for the stimulation pattern |

## Example Session

Builder detects 3 distinct colors in image → user assigns in session:

| Detected Color | User Label | Type | Amplitude | Duration |
|----------------|------------|------|-----------|----------|
| Group 1 | S1_pacing | Current injection | 50 μA | 2 ms |
| Group 2 | S2_premature | Current injection | 50 μA | 2 ms |
| Group 3 | Clamp_region | Voltage clamp | -80 mV | 100 ms |

*No predefined color mappings - colors are arbitrary based on user's input image.*

## Integration

StimBuilder is called by the root `Builder` coordinator after MeshBuilder completes. Output is saved in a format compatible with engine_V5.4.

## Status

*Architecture and protocols under development.*
