# The Silent Killer: Underlying Factors in Soccer Injuries

Behind the spectacle of goals and triumphant victories in European football, lie the silent tales of injuries, an understated yet influential player in the game. By applying machine learning to injury data, our investigation uncovers a hidden dimension to club success, revealing an often overlooked but crucially important aspect of the beautiful game.

Motivation for this project is highly inspired by the movie [Money Ball](<https://en.wikipedia.org/wiki/Moneyball_(film)>) and my own love for Football. Visca Barca! 🔴🔵

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Introduction](#introduction)
  - [Methodologies used](#methodologies-used)
  - [Tech Stack](#tech-stack)
- [Data Collection](#data-collection)
- [Data and Model Engineering](#data-and-model-engineering)
- [Model Building](#model-building)
- [Results](#results)
- [Discussion and Future Steps](#discussion-and-future-steps)

## Introduction

Brazil, favourites to win the 2014 FIFA World Cup, saw their prospects diminish following the injuries of star player Neymar. With their subsequent 7-1 loss to Germany, Brazil became a key example of how injuries can alter the outcome of a match and dictate team success. This investigations primary aim is to create predictive supervised learning models to assess the risk of injuries in soccer players, using binary classification. These predictive models could provide valuable insights for coaching and management staff, enabling them to mitigate injury risks by adjusting player workload, increasing physiotherapy, or making informed decisions during player signings. The overall goal is to decrease the number of injuries within a team, allowing players to maintain optimal performance throughout the season. As a secondary objective, the project will also strive to uncover and understand the underlying factors that either contribute to, or protect against, these injuries.

![Project Image](https://www.elfutbolero.us/__export/1632854462101/sites/elfutboleromx/img/2021/09/28/neymar.jpg_1635092770.jpg)

### Methodologies used

- Web Scraping
- Supervised Machine Learning
- Data Engineering
- Inferential Statistics
- Data Visualization

### Tech Stack

The following libraries are used in this project:

- **Pandas and Numpy**: These libraries are fundamental for data manipulation and analysis.

- **Matplotlib**: Used for data visualization to create insightful charts and plots.

- **TensorFlow and Keras**: These two powerful libraries are used for creating and training the Neural Network used in the project.

- **Scikit-Learn**: A comprehensive machine learning library used in this project for tasks such as training classifiers, model selection, and preprocessing.

- **Ast**: Used for working with Python's abstract syntax trees, which are crucial for certain data manipulation tasks.

- **OS and Warnings**: Used for interacting with the operating system and handling warnings, ensuring smooth execution of the code.

- **Datetime**: An essential library for manipulating dates and times, crucial for certain data preprocessing tasks.

- **Re**: Used for working with regular expressions, allowing for advanced string manipulation.

- **Requests and BeautifulSoup4**: These libraries are used for web scraping tasks, to collect the necessary data for analysis.

## Data Collection

In order to build effective models, a comprehensive dataset scraped from Transfermarkt and FBREF was necessary. This dataset included a diverse range of biometric and in-game statistical data for soccer players competing in Europe's top 5 leagues. The target feature was the player's current injury status.

Our preparation process was thorough, with meticulous data cleaning and pre-processing undertaken to ensure the dataset was fit for purpose. This included unpacking arrays, manipulating strings, encoding categorical data, and addressing missing values.

After eliminating duplicate entries and resolving inconsistencies in player names across the two sources, we compiled a robust, cleaned dataset of 1922 entries. Each entry was one-hot-encoded into numeric types to ensure the data was ready for use in our predictive models.

## Data and Model Engineering

in progress...

## Model Building

in progress...

## Results

in progress...

## Discussion and Future Steps

in progress...
