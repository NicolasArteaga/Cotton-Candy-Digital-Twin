# Cotton Candy Digital Twin - Feature Documentation

This document provides a comprehensive overview of all 29 features extracted from the cotton candy manufacturing process for decision tree modeling, along with the quality target metrics.

## Table of Contents
- [Overview](#overview)
- [Feature Categories](#feature-categories)
- [Core Process Parameters](#core-process-parameters)
- [Timing Metrics](#timing-metrics)
- [Environmental Baseline](#environmental-baseline)
- [Internal Environmental Sensors](#internal-environmental-sensors)
- [Quality Target Metrics](#quality-target-metrics)
- [Feature Usage Guidelines](#feature-usage-guidelines)

## Overview

The Cotton Candy Digital Twin extracts **29 features** from manufacturing process logs to predict and optimize cotton candy quality. These features capture:

- **Process control parameters** (user-configurable settings)
- **Timing characteristics** (process dynamics and efficiency)
- **Environmental conditions** (external baseline and internal sensor dynamics)

The features are designed to enable machine learning models to predict quality outcomes and optimize manufacturing parameters.

---

## Feature Categories

### 1. Core Process Parameters (4 features)

These are the primary control parameters that operators can adjust to influence the cotton candy manufacturing process.

#### `iteration_since_maintenance`
- **Type**: Integer
- **Unit**: Count
- **Description**: Number of cotton candy production cycles completed since the last maintenance operation
- **Impact**: Higher values may indicate unclean machine, potentially affecting quality and consistency
- **Range**: 0 to 60
- **Example**: 0 (freshly maintained), 30 (needs maintenance soon)

#### `wait_time`
- **Type**: Float
- **Unit**: Seconds
- **Description**: Pre-production waiting period before the cotton candy starts flowing
- **Typical Range**: 30-110 seconds
- **Example**: 102 seconds

#### `cook_time`
- **Type**: Float
- **Unit**: Seconds
- **Description**: Duration of the heating/cooking phase where sugar is melted and prepared for spinning
- **Impact**: Critical for sugar melting consistency, directly affects cotton candy texture and formation
- **Typical Range**: 30-115 seconds
- **Example**: 105 seconds

#### `cooldown_time`
- **Type**: Float
- **Unit**: Seconds
- **Description**: Cooling period after production of cc, so that machine can start another cc in normal temperature
- **Typical Range**: 30-120 seconds
- **Example**: 60 seconds

---

### 2. Timing Metrics (3 features)

These features capture the temporal dynamics of the manufacturing process and operational efficiency.

#### `duration_till_handover`
- **Type**: Float
- **Unit**: Seconds
- **Description**: Total time from process start until the cotton candy is handed over
- **Impact**: Indicates overall process efficiency and customer waiting time
- **Components**: Includes wait_time, cook_time, production time, and preparation for handover
- **Example**: 293.13 seconds (~4.9 minutes)

#### `duration_total`
- **Type**: Float
- **Unit**: Seconds
- **Description**: Complete process duration from start to finish, including all phases
- **Impact**: Overall process efficiency metric, affects throughput and energy consumption
- **Components**: All process phases including post-handover activities
- **Example**: 503.94 seconds (~8.4 minutes)

#### `duration_cc_flow`
- **Type**: Float
- **Unit**: Seconds
- **Description**: Active cotton candy production time - from "Show Start" to "Show End"
- **Impact**: Actual production time when cotton candy is being formed and flowing from the machine
- **Quality Relevance**: shorter flow times may affect consistency; optimal duration ensures proper formation
- **Example**: 106.63 seconds (~1.8 minutes)

---

### 3. Environmental Baseline (2 features)

External environmental conditions measured at the beginning of the process to establish baseline conditions.

#### `baseline_env_EnvH`
- **Type**: Float
- **Unit**: Percentage (%)
- **Description**: External environmental humidity at process start
- **Impact**: High humidity can affect sugar crystallization and cotton candy texture
- **Optimal Range**: 40-60% (varies by recipe)
- **Measurement Point**: Before machine turn-on phase

#### `baseline_env_EnvT`
- **Type**: Float
- **Unit**: Degrees Celsius (°C)
- **Description**: External environmental temperature at process start
- **Impact**: Ambient temperature affects cooling rates and final product characteristics
- **Optimal Range**: 18-25°C (varies by season and recipe)
- **Measurement Point**: Before machine turn-on phase

---

### 4. Internal Environmental Sensors (20 features)

Internal machine sensors monitored across 5 critical process phases. Each phase captures 4 sensor readings:

- **InH**: Internal Humidity (%)
- **InT**: Internal Temperature (°C)  
- **IrO**: Infrared Object/Head temperature (°C)
- **IrA**: Infrared Ambient temperature (°C)

#### Phase 1: Before Turn On (`before_turn_on_*`)

**Measurement Point**: Before the machine and heat is activated
**Purpose**: Establish internal baseline conditions

- `before_turn_on_env_InH`: Internal humidity before activation
- `before_turn_on_env_InT`: Internal temperature before activation
- `before_turn_on_env_IrO`: Infrared head temperature before activation
- `before_turn_on_env_IrA`: Infrared ambient temperature before activation

#### Phase 2: After Flow Start (`after_flow_start_*`)

**Measurement Point**: Immediately after cotton candy begins flowing
**Purpose**: Capture initial production conditions

- `after_flow_start_env_InH`: Internal humidity during early production
- `after_flow_start_env_InT`: Internal temperature during early production
- `after_flow_start_env_IrO`: Infrared head temperature during early production
- `after_flow_start_env_IrA`: Infrared ambient temperature during early production

#### Phase 3: After Flow End (`after_flow_end_*`)

**Measurement Point**: Immediately after cotton candy stops flowing
**Purpose**: Capture end-of-production conditions

- `after_flow_end_env_InH`: Internal humidity at production end
- `after_flow_end_env_InT`: Internal temperature at production end
- `after_flow_end_env_IrO`: Infrared head temperature at production end
- `after_flow_end_env_IrA`: Infrared ambient temperature at production end

#### Phase 4: Before Cooldown Process Start (`before_cooldown_*`)

**Measurement Point**: After weigh_place is complete, before the cooldown process starts
**Purpose**: Capture conditions before the Cooldown Process Start

- `before_cooldown_env_InH`: Internal humidity before cooldown process starts
- `before_cooldown_env_InT`: Internal temperature before cooldown process starts
- `before_cooldown_env_IrO`: Infrared head temperature before cooldown process starts
- `before_cooldown_env_IrA`: Infrared ambient temperature before cooldown process starts

#### Phase 5: After Cooldown Process End (`after_cooldown_*`)

**Measurement Point**: At the very end of the complete process, after cooldown is finished
**Purpose**: Capture final system state to compare to before the cooldown process started

- `after_cooldown_env_InH`: Final internal humidity after cooldown
- `after_cooldown_env_InT`: Final internal temperature after cooldown
- `after_cooldown_env_IrO`: Final infrared head temperature after cooldown
- `after_cooldown_env_IrA`: Final infrared ambient temperature after cooldown

---

iteration_since_maintenance,wait_time,cook_time,cooldown_time,duration_till_handover,duration_total,duration_cc_flow,baseline_env_EnvH,baseline_env_EnvT,before_turn_on_env_InH,before_turn_on_env_InT,before_turn_on_env_IrO,before_turn_on_env_IrA,after_flow_start_env_InH,after_flow_start_env_InT,after_flow_start_env_IrO,after_flow_start_env_IrA,after_flow_end_env_InH,after_flow_end_env_InT,after_flow_end_env_IrO,after_flow_end_env_IrA,before_cooldown_env_InH,before_cooldown_env_InT,before_cooldown_env_IrO,before_cooldown_env_IrA,after_cooldown_env_InH,after_cooldown_env_InT,after_cooldown_env_IrO,after_cooldown_env_IrA
0,102,105,60,293.130833,503.936259,106.631467,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0

