---
title: "TeaInventory - Beverage Shop Inventory Management System"
date: 2025-04-03T10:00:00+08:00
draft: false
math: true
summary: "An intelligent inventory management system designed for beverage shops, applying dynamic safety stock theory and machine learning to optimize restocking decisions."
tags: ["iOS", "Swift", "CoreData", "Machine Learning", "Inventory Management"]
categories: ["Projects", "Mobile Development"]
---

# TeaInventory: Beverage Shop Inventory Management System - Technical Documentation

## 1. Project Overview

TeaInventory is an iOS inventory management system specifically designed for beverage shops, developed using SwiftUI and CoreData. It addresses key challenges in inventory management, ingredient consumption prediction, and intelligent restocking for tea shops. The project implements the MVVM architecture combined with machine learning technology to achieve intelligent inventory management.

## 2. Technical Architecture

### 2.1 Development Environment
- Language: Swift 5.9+
- UI Framework: SwiftUI
- Data Persistence: CoreData
- Charts: SwiftUI Charts
- Minimum iOS Version: iOS 16.0

### 2.2 Architecture Design
The project uses the MVVM (Model-View-ViewModel) architecture:
- **Model**: CoreData entities and business logic classes
- **View**: SwiftUI view layer
- **ViewModel**: Logic controllers, such as InventoryManager

## 3. Core Data Models

The project uses CoreData to design the following main entities:

- **Ingredient**: Contains attributes like name, category, currentStock, initialStock, price, leadTime, etc.
- **Product**: Contains name, price, isActive attributes and associations with RecipeItem and SalesRecord
- **RecipeItem**: Intermediate entity connecting Product and Ingredient
- **SalesRecord**: Records product sales data
- **RestockRecord**: Records ingredient restocking history
- **InventoryCount**: Records inventory counting data

## 4. Core Functional Modules

### 4.1 Ingredient Management (IngredientView)
- Ingredient CRUD operations
- Category management and filtering
- Detailed information display and editing
- Inventory status visualization (red-yellow-green indicators)
- Ingredient deletion functionality (with cascading deletion of related records)

### 4.2 Product Management (ProductView)
- Product CRUD operations
- Recipe editing
- Sales records
- Product detail view
- Product deletion functionality (clearing related recipes and sales records)

### 4.3 Inventory Count (InventoryCountView)
- Real-time inventory counting
- Count history records
- Calculation of actual consumption based on count differences
- Batch counting functionality (renamed to "Inventory Adjustment")

### 4.4 Restock Management (RestockRecordView)
- Intelligent restocking suggestions
- Restock record management
- One-click restocking functionality
- Restock quantity calculation based on dynamic safety stock

### 4.5 Data Analysis (AnalyticsView)
- Sales trend analysis
- Ingredient consumption analysis
- Chart visualization

### 4.6 Settings Management (SettingsView)
- Feature toggle controls
- Prediction model parameter adjustments
- Regression model parameter adjustments
- Data import/export
- System reset functionality

## 5. Intelligent Prediction System

### 5.1 Basic Prediction Model: Dynamic Safety Stock Calculation

#### 5.1.1 Mathematical Principles

The basic prediction model is primarily based on **dynamic safety stock theory**, using exponential smoothing and service level concepts to optimize inventory management.

The core formula for safety stock calculation is:

$$SS = z \cdot \sigma_L \cdot \sqrt{L}$$

Where:
- $SS$ is the safety stock level
- $z$ is the Z-score corresponding to the service level in standard normal distribution
- $\sigma_L$ is the standard deviation of demand
- $L$ is the lead time for restocking

In the system implementation, it is expressed as:

$$SS = \overline{D} \cdot L + z \cdot \sigma_D \cdot \sqrt{L}$$

Where:
- $\overline{D}$ is the daily average consumption
- $\sigma_D$ is the standard deviation of consumption

#### 5.1.2 Service Level and Z-Score Relationship

Service levels and Z-scores are related through the standard normal distribution function:

| Service Level | Z-Score |
|--------------|---------|
| 90%          | 1.28    |
| 95%          | 1.65    |
| 98%          | 2.33    |
| 99%          | 2.58    |

#### 5.1.3 Exponential Smoothing

Exponential smoothing is used to predict future consumption, calculated as:

$$F_t = \alpha \cdot D_{t-1} + (1-\alpha) \cdot F_{t-1}$$

Where:
- $F_t$ is the forecast for time t
- $D_{t-1}$ is the actual consumption at time t-1
- $F_{t-1}$ is the forecast for time t-1
- $\alpha$ is the smoothing coefficient (0.1-0.5)

### 5.1.4 Code Implementation

```swift
func calculateDynamicSafetyStock(ingredient: Ingredient, alpha: Double? = nil, serviceLevel: Double? = nil) -> Double {
    // If prediction feature is turned off, return a percentage of initial stock as safety stock
    if !enablePrediction {
        return ingredient.initialStock * 0.2
    }
    
    let serviceLevelValue = serviceLevel ?? self.serviceLevel
    
    let dailyConsumptionAverage = calculateDailyConsumptionAverage(ingredient: ingredient)
    let standardDeviation = calculateConsumptionStandardDeviation(ingredient: ingredient)
    let leadTime = Double(ingredient.leadTime)
    
    // Service level coefficient (Z)
    let zScore = calculateZScore(serviceLevel: serviceLevelValue)
    
    return dailyConsumptionAverage * leadTime + zScore * standardDeviation * sqrt(leadTime)
}

// Z-score calculation
private func calculateZScore(serviceLevel: Double) -> Double {
    switch serviceLevel {
    case 0.90...0.91: return 1.28
    case 0.92...0.93: return 1.41
    case 0.94...0.95: return 1.65
    case 0.96...0.97: return 1.88
    case 0.98...0.99: return 2.33
    case 0.9985...0.9999: return 3.0
    default: return 1.65
    }
}

// Standard deviation calculation
private func calculateStandardDeviation(values: [Double]) -> Double {
    guard !values.isEmpty else { return 0.0 }
    
    let count = Double(values.count)
    let mean = values.reduce(0.0, +) / count
    let sumOfSquaredDifferences = values.map { pow($0 - mean, 2) }.reduce(0.0, +)
    
    return sqrt(sumOfSquaredDifferences / count)
}
```

### 5.1.5 Reorder Point and Restock Quantity Calculation

The system determines whether restocking is needed based on current inventory and safety stock:

$$RestockQuantity = (DailyConsumptionAverage \times LeadTime + SafetyStock) - CurrentStock$$

This is implemented in code as:

```swift
func calculateSuggestedRestockAmount(ingredient: Ingredient) -> Double {
    let remainingStock = calculateRemainingStock(ingredient: ingredient)
    
    if enablePrediction {
        let dailyConsumptionAverage = calculateDailyConsumptionAverage(ingredient: ingredient)
        let leadTime = Double(ingredient.leadTime)
        
        // Calculate expected consumption during lead time
        let expectedConsumptionDuringLeadTime = dailyConsumptionAverage * leadTime
        
        // Calculate safety stock
        let safetyStock = calculateDynamicSafetyStock(ingredient: ingredient)
        
        // Suggested restock amount = Expected consumption during lead time + Safety stock - Current stock
        let suggestedAmount = expectedConsumptionDuringLeadTime + safetyStock - remainingStock
        
        return max(0, suggestedAmount)
    } else {
        return max(0, ingredient.initialStock - remainingStock)
    }
}
```

### 5.2 Regression Analysis Model: Ingredient Consumption Prediction

#### 5.2.1 Mathematical Principles

The regression analysis model uses **multivariate linear regression** techniques to establish the relationship between product sales and ingredient consumption. The core is solving the matrix equation:

$$\mathbf{y} = \mathbf{X}\mathbf{\beta} + \mathbf{\epsilon}$$

Where:
- $\mathbf{y}$ is the ingredient consumption vector
- $\mathbf{X}$ is the product sales matrix
- $\mathbf{\beta}$ is the regression coefficient vector (to be determined)
- $\mathbf{\epsilon}$ is the error term

The system implements linear regression with L2 regularization (Ridge regression), with the objective function:

$$J(\mathbf{\beta}) = \frac{1}{2m}\sum_{i=1}^{m}(h_{\mathbf{\beta}}(\mathbf{x}^{(i)}) - y^{(i)})^2 + \frac{\lambda}{2m}\sum_{j=1}^{n}\beta_j^2$$

Where:
- $m$ is the number of samples
- $n$ is the number of features (product types)
- $\lambda$ is the regularization parameter
- $h_{\mathbf{\beta}}(\mathbf{x}^{(i)})$ is the prediction function, equal to $\mathbf{x}^{(i)} \cdot \mathbf{\beta}$

#### 5.2.2 Hybrid Prediction System

The final prediction uses a hybrid system, combining recipe calculation and regression prediction:

$$Consumption = \alpha \times RecipeCalculatedValue + (1-\alpha) \times RegressionPredictedValue$$

Where $\alpha$ is adjusted with model maturity:

$$\alpha = 1 - ModelMaturity$$

#### 5.2.3 Gradient Descent Training

```swift
// Train model using gradient descent algorithm
private func trainWithGradientDescent(X: [[Double]], y: [Double], iterations: Int = 1000, learningRate: Double = 0.01, l2Regularization: Double = 0.1) -> [Double] {
    let m = Double(X.count)  // Number of samples
    let n = X[0].count       // Number of features
    
    // Initialize parameters
    var theta = Array(repeating: 0.0, n)
    
    // Execute gradient descent
    for _ in 0..<iterations {
        // Calculate predictions
        let predictions = X.map { row in
            dotProduct(row, theta)
        }
        
        // Calculate errors
        let errors = zip(predictions, y).map { $0 - $1 }
        
        // Update parameters (with L2 regularization)
        for j in 0..<n {
            let gradient = (1.0/m) * (
                zip(errors, X).map { error, row in
                    error * row[j]
                }.reduce(0, +) +
                l2Regularization * theta[j]  // L2 regularization term
            )
            
            theta[j] = theta[j] - learningRate * gradient
        }
    }
    
    return theta
}

// Calculate dot product
private func dotProduct(_ row: [Double], _ theta: [Double]) -> Double {
    return zip(row, theta).reduce(0.0) { $0 + $1.0 * $1.1 }
}
```

#### 5.2.4 Feature Normalization

```swift
// Normalize features
private func normalizeFeatures(X: [[Double]]) -> ([[Double]], [Double], [Double]) {
    let n = X[0].count
    var means = Array(repeating: 0.0, count: n)
    var stdDevs = Array(repeating: 0.0, count: n)
    
    // Calculate mean for each column
    for j in 0..<n {
        let column = X.map { $0[j] }
        means[j] = column.reduce(0.0, +) / Double(column.count)
    }
    
    // Calculate standard deviation for each column
    for j in 0..<n {
        let column = X.map { $0[j] }
        let squaredDiffs = column.map { pow($0 - means[j], 2) }
        stdDevs[j] = sqrt(squaredDiffs.reduce(0.0, +) / Double(column.count))
        // Prevent division by zero
        if stdDevs[j] == 0 {
            stdDevs[j] = 1.0
        }
    }
    
    // Normalize data
    let normalizedX = X.map { row in
        (0..<n).map { j in
            (row[j] - means[j]) / stdDevs[j]
        }
    }
    
    return (normalizedX, means, stdDevs)
}
```

#### 5.2.5 Hybrid Prediction Implementation

```swift
// Predict consumption
func predictConsumption(ingredientID: String, sales: [String: Double], recipeBasedConsumption: Double) -> Double {
    // Get hybrid coefficient
    let alpha = RegressionModelStore.shared.getModelAlpha()
    
    // Recipe-based calculation
    let recipeAmount = recipeBasedConsumption
    
    // If no trained model or empty sales data, return recipe-based calculation
    guard let model = models[ingredientID], !sales.isEmpty else {
        return recipeAmount
    }
    
    // Prepare feature vector
    var features: [Double] = []
    for productID in model.productIDs {
        features.append(sales[productID] ?? 0.0)
    }
    
    // Feature normalization
    let normalizedFeatures = features.enumerated().map { j, value in
        (value - model.means[j]) / model.stdDevs[j]
    }
    
    // Calculate prediction
    let prediction = dotProduct(normalizedFeatures, model.coefficients)
    
    // Use hybrid model
    return alpha * recipeAmount + (1 - alpha) * prediction
}
```

#### 5.2.6 Model Maturity Calculation

Model maturity is calculated based on the number of training samples collected and prediction accuracy:

```swift
func getRegressionModelMaturity() -> Double {
    // Sample ratio, maximum 1.0
    let sampleRatio = min(Double(collectedDataPoints) / Double(requiredDataPointsForMaturity), 1.0)
    
    // MSE change ratio, maximum 1.0
    let mseRatio = baselineMSE > 0 ? min(1.0 - (currentMSE / baselineMSE), 1.0) : 0.0
    
    // Combined score, sample size has 70% weight, error improvement has 30% weight
    let maturity = 0.7 * sampleRatio + 0.3 * mseRatio
    
    return max(0.0, min(maturity, 1.0))
}
```

### 5.3 Model Integration and Adaptive Mechanisms

The Tea system integrates the two models and implements adaptive optimization through the following mechanisms:

1. **Data Collection**: Collecting actual consumption data through inventory count differences and sales records

2. **Model Training Triggers**:
   - Early versions: New data automatically triggers training
   - Current version: Manual confirmation of training to ensure high-quality data

3. **Adaptive Adjustment of Hybrid Coefficients**:
   - As model maturity increases, α value automatically decreases
   - Users can manually adjust to accommodate recipe changes

4. **Prediction Evaluation**:
   - Using Mean Squared Error (MSE) to evaluate model performance
   - Normalized scoring by comparison with benchmark models

This integrated design combines classical inventory theory with machine learning methods to achieve efficient and accurate inventory prediction and management, particularly suitable for beverage shops with complex recipes and variable consumption patterns.

## 6. Technical Highlights

### 6.1 Optimized CoreData Usage
- Batch deletion operations (NSBatchDeleteRequest)
- Well-designed relationship models
- Cascading deletion implementation

### 6.2 Advanced SwiftUI Features
- Custom Alert and Sheet interactions
- Complex form design
- Dynamic lists and filtering
- Multi-level navigation structure
- Conditional rendering and state management

### 6.3 Advanced Error Prevention Mechanisms
- Multi-layer confirmation dialogs
- Text validation mechanisms
- Clear UI feedback for states

### 6.4 Data Analysis and Visualization
- Trend charts
- Inventory status visualization
- Intelligent restocking suggestions

### 6.5 Google Sheets Integration
- Cloud data synchronization via GoogleSheetsManager
- Support for various operation logs

## 7. Security Features

- Sensitive operations (such as deletion and reset) require multiple confirmations
- Data validation and error handling
- Operation logging

## 8. Extensibility Design

The system adopts a modular design for easy extension of new features:
- Reserved data import/export interfaces
- Configurable prediction parameters
- Customizable UI components
- Loosely coupled business logic

## 9. Future Development Directions

- Further optimization of regression model algorithms
- Addition of more data analysis features
- Implementation of complete data import/export functionality
- Support for multi-user collaboration and permission management
- Enhanced integration with external systems (such as POS)

The TeaInventory system significantly improves beverage shop inventory management efficiency through advanced prediction algorithms and intuitive user interfaces, reducing inventory waste and optimizing the restocking process, representing the latest level of inventory management technology in the industry.