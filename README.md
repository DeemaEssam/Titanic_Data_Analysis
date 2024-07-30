# Titanic


{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "# Titanic - Machine Learning"
      ],
      "metadata": {
        "id": "fJllAEdnLzHY"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "This project aims to introduce the most important steps of data analysis and explore the different stages. We will use the data of Titanic survivors available on the Kaggle website at the following link:\n",
        "https://www.kaggle.com/competitions/titanic/overview\n",
        "\n",
        "You can download the dataset and explore all the information about it in the following link: https://www.kaggle.com/competitions/titanic/data"
      ],
      "metadata": {
        "id": "23qsq5sDL_nC"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Note: To complete this projct, you need to modify the cells that contain the code below before submitting the project. All other cells should remain unchanged without any modifications.\n",
        "\n",
        "############################ <br>\n",
        "أكمل الكود <br>\n",
        "Complete the code <br>\n",
        "############################ <br>"
      ],
      "metadata": {
        "id": "p-v5KaJZPbvm"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Importing the Dependencies"
      ],
      "metadata": {
        "id": "E86z4W-fNRkA"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.metrics import accuracy_score"
      ],
      "metadata": {
        "id": "3UV9a52tNPWP"
      },
      "execution_count": 94,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Reading the data\n"
      ],
      "metadata": {
        "id": "Imj-Udf5N5k_"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "We will first use `pd.read_csv` to load the data from csv file to Pandas DataFrame:"
      ],
      "metadata": {
        "id": "94vLT17oOMfe"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "data = pd.read_csv('/content/train.csv')"
      ],
      "metadata": {
        "id": "KXw6K3tTOL_h"
      },
      "execution_count": 95,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "After reading the data, we will now review the data to ensure it has been read correctly by using the command `head`:"
      ],
      "metadata": {
        "id": "mm3GuQCcOUzG"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "#this will print first 5 rows in the dataset\n",
        "data.head()"
      ],
      "metadata": {
        "id": "eyqBM9dhOhAY",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "outputId": "b6e88705-0667-4535-aa20-084cf01de27f"
      },
      "execution_count": 96,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   PassengerId  Survived  Pclass  \\\n",
              "0            1         0       3   \n",
              "1            2         1       1   \n",
              "2            3         1       3   \n",
              "3            4         1       1   \n",
              "4            5         0       3   \n",
              "\n",
              "                                                Name     Sex   Age  SibSp  \\\n",
              "0                            Braund, Mr. Owen Harris    male  22.0      1   \n",
              "1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   \n",
              "2                             Heikkinen, Miss. Laina  female  26.0      0   \n",
              "3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   \n",
              "4                           Allen, Mr. William Henry    male  35.0      0   \n",
              "\n",
              "   Parch            Ticket     Fare Cabin Embarked  \n",
              "0      0         A/5 21171   7.2500   NaN        S  \n",
              "1      0          PC 17599  71.2833   C85        C  \n",
              "2      0  STON/O2. 3101282   7.9250   NaN        S  \n",
              "3      0            113803  53.1000  C123        S  \n",
              "4      0            373450   8.0500   NaN        S  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-d7e19c3d-91f3-4b3e-925f-afd5dfc72de5\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>PassengerId</th>\n",
              "      <th>Survived</th>\n",
              "      <th>Pclass</th>\n",
              "      <th>Name</th>\n",
              "      <th>Sex</th>\n",
              "      <th>Age</th>\n",
              "      <th>SibSp</th>\n",
              "      <th>Parch</th>\n",
              "      <th>Ticket</th>\n",
              "      <th>Fare</th>\n",
              "      <th>Cabin</th>\n",
              "      <th>Embarked</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>Braund, Mr. Owen Harris</td>\n",
              "      <td>male</td>\n",
              "      <td>22.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>A/5 21171</td>\n",
              "      <td>7.2500</td>\n",
              "      <td>NaN</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
              "      <td>female</td>\n",
              "      <td>38.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>PC 17599</td>\n",
              "      <td>71.2833</td>\n",
              "      <td>C85</td>\n",
              "      <td>C</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>3</td>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>Heikkinen, Miss. Laina</td>\n",
              "      <td>female</td>\n",
              "      <td>26.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>STON/O2. 3101282</td>\n",
              "      <td>7.9250</td>\n",
              "      <td>NaN</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>4</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
              "      <td>female</td>\n",
              "      <td>35.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>113803</td>\n",
              "      <td>53.1000</td>\n",
              "      <td>C123</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>5</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>Allen, Mr. William Henry</td>\n",
              "      <td>male</td>\n",
              "      <td>35.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>373450</td>\n",
              "      <td>8.0500</td>\n",
              "      <td>NaN</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-d7e19c3d-91f3-4b3e-925f-afd5dfc72de5')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-d7e19c3d-91f3-4b3e-925f-afd5dfc72de5 button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-d7e19c3d-91f3-4b3e-925f-afd5dfc72de5');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "<div id=\"df-3208fbef-df6f-4bed-88cf-c33b2681e2d5\">\n",
              "  <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-3208fbef-df6f-4bed-88cf-c33b2681e2d5')\"\n",
              "            title=\"Suggest charts\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "  </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "      --bg-color: #E8F0FE;\n",
              "      --fill-color: #1967D2;\n",
              "      --hover-bg-color: #E2EBFA;\n",
              "      --hover-fill-color: #174EA6;\n",
              "      --disabled-fill-color: #AAA;\n",
              "      --disabled-bg-color: #DDD;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "      --bg-color: #3B4455;\n",
              "      --fill-color: #D2E3FC;\n",
              "      --hover-bg-color: #434B5C;\n",
              "      --hover-fill-color: #FFFFFF;\n",
              "      --disabled-bg-color: #3B4455;\n",
              "      --disabled-fill-color: #666;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart {\n",
              "    background-color: var(--bg-color);\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: var(--fill-color);\n",
              "    height: 32px;\n",
              "    padding: 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: var(--hover-bg-color);\n",
              "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: var(--button-hover-fill-color);\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart-complete:disabled,\n",
              "  .colab-df-quickchart-complete:disabled:hover {\n",
              "    background-color: var(--disabled-bg-color);\n",
              "    fill: var(--disabled-fill-color);\n",
              "    box-shadow: none;\n",
              "  }\n",
              "\n",
              "  .colab-df-spinner {\n",
              "    border: 2px solid var(--fill-color);\n",
              "    border-color: transparent;\n",
              "    border-bottom-color: var(--fill-color);\n",
              "    animation:\n",
              "      spin 1s steps(1) infinite;\n",
              "  }\n",
              "\n",
              "  @keyframes spin {\n",
              "    0% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "      border-left-color: var(--fill-color);\n",
              "    }\n",
              "    20% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    30% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    40% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    60% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    80% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "    90% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "  }\n",
              "</style>\n",
              "\n",
              "  <script>\n",
              "    async function quickchart(key) {\n",
              "      const quickchartButtonEl =\n",
              "        document.querySelector('#' + key + ' button');\n",
              "      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
              "      quickchartButtonEl.classList.add('colab-df-spinner');\n",
              "      try {\n",
              "        const charts = await google.colab.kernel.invokeFunction(\n",
              "            'suggestCharts', [key], {});\n",
              "      } catch (error) {\n",
              "        console.error('Error during call to suggestCharts:', error);\n",
              "      }\n",
              "      quickchartButtonEl.classList.remove('colab-df-spinner');\n",
              "      quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
              "    }\n",
              "    (() => {\n",
              "      let quickchartButtonEl =\n",
              "        document.querySelector('#df-3208fbef-df6f-4bed-88cf-c33b2681e2d5 button');\n",
              "      quickchartButtonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "    })();\n",
              "  </script>\n",
              "</div>\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "dataframe",
              "variable_name": "data",
              "summary": "{\n  \"name\": \"data\",\n  \"rows\": 891,\n  \"fields\": [\n    {\n      \"column\": \"PassengerId\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 257,\n        \"min\": 1,\n        \"max\": 891,\n        \"num_unique_values\": 891,\n        \"samples\": [\n          710,\n          440,\n          841\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Survived\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 1,\n        \"num_unique_values\": 2,\n        \"samples\": [\n          1,\n          0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Pclass\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 1,\n        \"max\": 3,\n        \"num_unique_values\": 3,\n        \"samples\": [\n          3,\n          1\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Name\",\n      \"properties\": {\n        \"dtype\": \"string\",\n        \"num_unique_values\": 891,\n        \"samples\": [\n          \"Moubarek, Master. Halim Gonios (\\\"William George\\\")\",\n          \"Kvillner, Mr. Johan Henrik Johannesson\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Sex\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 2,\n        \"samples\": [\n          \"female\",\n          \"male\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Age\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 14.526497332334042,\n        \"min\": 0.42,\n        \"max\": 80.0,\n        \"num_unique_values\": 88,\n        \"samples\": [\n          0.75,\n          22.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"SibSp\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 1,\n        \"min\": 0,\n        \"max\": 8,\n        \"num_unique_values\": 7,\n        \"samples\": [\n          1,\n          0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Parch\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 6,\n        \"num_unique_values\": 7,\n        \"samples\": [\n          0,\n          1\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Ticket\",\n      \"properties\": {\n        \"dtype\": \"string\",\n        \"num_unique_values\": 681,\n        \"samples\": [\n          \"11774\",\n          \"248740\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Fare\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 49.6934285971809,\n        \"min\": 0.0,\n        \"max\": 512.3292,\n        \"num_unique_values\": 248,\n        \"samples\": [\n          11.2417,\n          51.8625\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Cabin\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 147,\n        \"samples\": [\n          \"D45\",\n          \"B49\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Embarked\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 3,\n        \"samples\": [\n          \"S\",\n          \"C\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}"
            }
          },
          "metadata": {},
          "execution_count": 96
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# number of rows and columns\n",
        "data.shape"
      ],
      "metadata": {
        "id": "mkzj9Ye0q5e3",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "754051dc-3ff8-4aac-ed6e-c2c672988ba2"
      },
      "execution_count": 97,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(891, 12)"
            ]
          },
          "metadata": {},
          "execution_count": 97
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Data Preprocessing"
      ],
      "metadata": {
        "id": "tu1feHO3QjAq"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Now we will use the info command to learn more about the data, such as the number of rows and columns, data types, and the number of missing values."
      ],
      "metadata": {
        "id": "8k6Fg5dDRT68"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "data.info()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Gw5WmQf1Qgzx",
        "outputId": "84cc228c-17aa-489f-89b3-c0034fcb70bf"
      },
      "execution_count": 98,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 891 entries, 0 to 890\n",
            "Data columns (total 12 columns):\n",
            " #   Column       Non-Null Count  Dtype  \n",
            "---  ------       --------------  -----  \n",
            " 0   PassengerId  891 non-null    int64  \n",
            " 1   Survived     891 non-null    int64  \n",
            " 2   Pclass       891 non-null    int64  \n",
            " 3   Name         891 non-null    object \n",
            " 4   Sex          891 non-null    object \n",
            " 5   Age          714 non-null    float64\n",
            " 6   SibSp        891 non-null    int64  \n",
            " 7   Parch        891 non-null    int64  \n",
            " 8   Ticket       891 non-null    object \n",
            " 9   Fare         891 non-null    float64\n",
            " 10  Cabin        204 non-null    object \n",
            " 11  Embarked     889 non-null    object \n",
            "dtypes: float64(2), int64(5), object(5)\n",
            "memory usage: 83.7+ KB\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "### Dealing with Missing Data"
      ],
      "metadata": {
        "id": "eg8ZlJXydiGF"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# to view the Missing valuse in each column:\n",
        "data.isnull().sum()"
      ],
      "metadata": {
        "id": "MTmtZyO-eKeu",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "f1f03208-a0fd-47c1-d03a-0ed9ccf56372"
      },
      "execution_count": 99,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "PassengerId      0\n",
              "Survived         0\n",
              "Pclass           0\n",
              "Name             0\n",
              "Sex              0\n",
              "Age            177\n",
              "SibSp            0\n",
              "Parch            0\n",
              "Ticket           0\n",
              "Fare             0\n",
              "Cabin          687\n",
              "Embarked         2\n",
              "dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 99
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "You have three options to fix this:\n",
        "\n",
        "*   Delete rows that contains missing valuse\n",
        "*   Delete the whole column that contains missing values\n",
        "*   Replace missing values with some value (Mean, Median, Mode, constant)\n",
        "\n",
        "\n",
        "\n",
        "\n",
        "\n"
      ],
      "metadata": {
        "id": "QQkE4ciCgsTa"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "There are three columns contains Missing values: **Age, Cabin, Embarked**. <br>\n",
        "In the Age column, we will fill the missing values with the mean since it is a simple and quick method to handle missing data and helps maintain the overall distribution of the dataset."
      ],
      "metadata": {
        "id": "MLZUqOrxh7ok"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "############################\n",
        "#أكمل الكود\n",
        "#Complete the code\n",
        "############################\n",
        "\n",
        "#fill the missing values in Age with the mean of Age column\n",
        "#you can simply use 'filllna' function, or any other way such as SimpleImputer\n",
        "\n",
        "\n",
        "# Handling missing values\n",
        "for column in data:\n",
        "    if data[column].dtype in ['float64', 'int64']:\n",
        "        data[column].fillna(data[column].median(), inplace=True)\n",
        "    elif data[column].dtype == 'object':\n",
        "        data[column].fillna(data[column].mode()[0], inplace=True)\n"
      ],
      "metadata": {
        "id": "6AgyJ6DEjaKc"
      },
      "execution_count": 100,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "data['Age'].isnull().sum()"
      ],
      "metadata": {
        "id": "cBtBc-HUj28y",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "5f15a540-9d4e-4b11-9f56-6980e6808a4e"
      },
      "execution_count": 101,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0"
            ]
          },
          "metadata": {},
          "execution_count": 101
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "There are a large number of missing values in the Cabin column, so we will drop this column from the dataset."
      ],
      "metadata": {
        "id": "40pi536tkPPH"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "data = data.drop(['Cabin'], axis=1)"
      ],
      "metadata": {
        "id": "ip_B6VqOksUO"
      },
      "execution_count": 102,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "data.head()"
      ],
      "metadata": {
        "id": "DCpCC8ggk3nu",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "outputId": "0bba5be8-34db-4565-fd21-c71895e5a5a7"
      },
      "execution_count": 103,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   PassengerId  Survived  Pclass  \\\n",
              "0            1         0       3   \n",
              "1            2         1       1   \n",
              "2            3         1       3   \n",
              "3            4         1       1   \n",
              "4            5         0       3   \n",
              "\n",
              "                                                Name     Sex   Age  SibSp  \\\n",
              "0                            Braund, Mr. Owen Harris    male  22.0      1   \n",
              "1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   \n",
              "2                             Heikkinen, Miss. Laina  female  26.0      0   \n",
              "3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   \n",
              "4                           Allen, Mr. William Henry    male  35.0      0   \n",
              "\n",
              "   Parch            Ticket     Fare Embarked  \n",
              "0      0         A/5 21171   7.2500        S  \n",
              "1      0          PC 17599  71.2833        C  \n",
              "2      0  STON/O2. 3101282   7.9250        S  \n",
              "3      0            113803  53.1000        S  \n",
              "4      0            373450   8.0500        S  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-fdc28070-18ea-42bf-9fd1-103317f953e6\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>PassengerId</th>\n",
              "      <th>Survived</th>\n",
              "      <th>Pclass</th>\n",
              "      <th>Name</th>\n",
              "      <th>Sex</th>\n",
              "      <th>Age</th>\n",
              "      <th>SibSp</th>\n",
              "      <th>Parch</th>\n",
              "      <th>Ticket</th>\n",
              "      <th>Fare</th>\n",
              "      <th>Embarked</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>Braund, Mr. Owen Harris</td>\n",
              "      <td>male</td>\n",
              "      <td>22.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>A/5 21171</td>\n",
              "      <td>7.2500</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
              "      <td>female</td>\n",
              "      <td>38.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>PC 17599</td>\n",
              "      <td>71.2833</td>\n",
              "      <td>C</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>3</td>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>Heikkinen, Miss. Laina</td>\n",
              "      <td>female</td>\n",
              "      <td>26.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>STON/O2. 3101282</td>\n",
              "      <td>7.9250</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>4</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
              "      <td>female</td>\n",
              "      <td>35.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>113803</td>\n",
              "      <td>53.1000</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>5</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>Allen, Mr. William Henry</td>\n",
              "      <td>male</td>\n",
              "      <td>35.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>373450</td>\n",
              "      <td>8.0500</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-fdc28070-18ea-42bf-9fd1-103317f953e6')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-fdc28070-18ea-42bf-9fd1-103317f953e6 button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-fdc28070-18ea-42bf-9fd1-103317f953e6');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "<div id=\"df-e6bc688f-40b7-4590-bab2-1ec3634763f7\">\n",
              "  <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-e6bc688f-40b7-4590-bab2-1ec3634763f7')\"\n",
              "            title=\"Suggest charts\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "  </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "      --bg-color: #E8F0FE;\n",
              "      --fill-color: #1967D2;\n",
              "      --hover-bg-color: #E2EBFA;\n",
              "      --hover-fill-color: #174EA6;\n",
              "      --disabled-fill-color: #AAA;\n",
              "      --disabled-bg-color: #DDD;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "      --bg-color: #3B4455;\n",
              "      --fill-color: #D2E3FC;\n",
              "      --hover-bg-color: #434B5C;\n",
              "      --hover-fill-color: #FFFFFF;\n",
              "      --disabled-bg-color: #3B4455;\n",
              "      --disabled-fill-color: #666;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart {\n",
              "    background-color: var(--bg-color);\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: var(--fill-color);\n",
              "    height: 32px;\n",
              "    padding: 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: var(--hover-bg-color);\n",
              "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: var(--button-hover-fill-color);\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart-complete:disabled,\n",
              "  .colab-df-quickchart-complete:disabled:hover {\n",
              "    background-color: var(--disabled-bg-color);\n",
              "    fill: var(--disabled-fill-color);\n",
              "    box-shadow: none;\n",
              "  }\n",
              "\n",
              "  .colab-df-spinner {\n",
              "    border: 2px solid var(--fill-color);\n",
              "    border-color: transparent;\n",
              "    border-bottom-color: var(--fill-color);\n",
              "    animation:\n",
              "      spin 1s steps(1) infinite;\n",
              "  }\n",
              "\n",
              "  @keyframes spin {\n",
              "    0% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "      border-left-color: var(--fill-color);\n",
              "    }\n",
              "    20% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    30% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    40% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    60% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    80% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "    90% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "  }\n",
              "</style>\n",
              "\n",
              "  <script>\n",
              "    async function quickchart(key) {\n",
              "      const quickchartButtonEl =\n",
              "        document.querySelector('#' + key + ' button');\n",
              "      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
              "      quickchartButtonEl.classList.add('colab-df-spinner');\n",
              "      try {\n",
              "        const charts = await google.colab.kernel.invokeFunction(\n",
              "            'suggestCharts', [key], {});\n",
              "      } catch (error) {\n",
              "        console.error('Error during call to suggestCharts:', error);\n",
              "      }\n",
              "      quickchartButtonEl.classList.remove('colab-df-spinner');\n",
              "      quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
              "    }\n",
              "    (() => {\n",
              "      let quickchartButtonEl =\n",
              "        document.querySelector('#df-e6bc688f-40b7-4590-bab2-1ec3634763f7 button');\n",
              "      quickchartButtonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "    })();\n",
              "  </script>\n",
              "</div>\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "dataframe",
              "variable_name": "data",
              "summary": "{\n  \"name\": \"data\",\n  \"rows\": 891,\n  \"fields\": [\n    {\n      \"column\": \"PassengerId\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 257,\n        \"min\": 1,\n        \"max\": 891,\n        \"num_unique_values\": 891,\n        \"samples\": [\n          710,\n          440,\n          841\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Survived\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 1,\n        \"num_unique_values\": 2,\n        \"samples\": [\n          1,\n          0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Pclass\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 1,\n        \"max\": 3,\n        \"num_unique_values\": 3,\n        \"samples\": [\n          3,\n          1\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Name\",\n      \"properties\": {\n        \"dtype\": \"string\",\n        \"num_unique_values\": 891,\n        \"samples\": [\n          \"Moubarek, Master. Halim Gonios (\\\"William George\\\")\",\n          \"Kvillner, Mr. Johan Henrik Johannesson\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Sex\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 2,\n        \"samples\": [\n          \"female\",\n          \"male\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Age\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 13.019696550973201,\n        \"min\": 0.42,\n        \"max\": 80.0,\n        \"num_unique_values\": 88,\n        \"samples\": [\n          0.75,\n          22.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"SibSp\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 1,\n        \"min\": 0,\n        \"max\": 8,\n        \"num_unique_values\": 7,\n        \"samples\": [\n          1,\n          0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Parch\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 6,\n        \"num_unique_values\": 7,\n        \"samples\": [\n          0,\n          1\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Ticket\",\n      \"properties\": {\n        \"dtype\": \"string\",\n        \"num_unique_values\": 681,\n        \"samples\": [\n          \"11774\",\n          \"248740\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Fare\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 49.6934285971809,\n        \"min\": 0.0,\n        \"max\": 512.3292,\n        \"num_unique_values\": 248,\n        \"samples\": [\n          11.2417,\n          51.8625\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Embarked\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 3,\n        \"samples\": [\n          \"S\",\n          \"C\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}"
            }
          },
          "metadata": {},
          "execution_count": 103
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "In the Embarked column, there are only two missing values. Let's see what the categories in this column are."
      ],
      "metadata": {
        "id": "_1Xp5fmjk_kV"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "data['Embarked'].value_counts()"
      ],
      "metadata": {
        "id": "Apl5phzEmYjg",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "4278f1ea-f236-45c9-de39-a459fbec43a3"
      },
      "execution_count": 104,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Embarked\n",
              "S    646\n",
              "C    168\n",
              "Q     77\n",
              "Name: count, dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 104
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "############################\n",
        "#أكمل الكود\n",
        "#Complete the code\n",
        "############################\n",
        "\n",
        "#fill the missing values in Embarked with the mode of Embarked column:\n",
        "\n",
        "\n",
        "# Handling missing values\n",
        "for Embarked in data:\n",
        "    if data[Embarked].dtype in ['float64', 'int64']:\n",
        "        data[Embarked].fillna(data[Embarked].median(), inplace=True)\n",
        "    elif data[Embarked].dtype == 'object':\n",
        "        data[Embarked].fillna(data[Embarked].mode()[0], inplace=True)\n",
        "\n"
      ],
      "metadata": {
        "id": "uEcN0y09lZmF"
      },
      "execution_count": 105,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "data['Embarked'].isnull().sum()"
      ],
      "metadata": {
        "id": "RyYeYgfGm5WE",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "c9a653ca-c665-4509-fa20-7ae287ae8667"
      },
      "execution_count": 106,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0"
            ]
          },
          "metadata": {},
          "execution_count": 106
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "### Drop useless columns"
      ],
      "metadata": {
        "id": "5eIYHEeXdrRJ"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "As you know, the PassengerId and Name of the Passenger do not affect the probability of survival. and ticket column does not have a clear relationship to the survival of passengers, so they will be dropped:"
      ],
      "metadata": {
        "id": "X033fm49eRXv"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "############################\n",
        "#أكمل الكود\n",
        "#Complete the code\n",
        "############################\n",
        "\n",
        "# Drop the PassengerId and Name Columns from the dataset:\n",
        "\n",
        "data = data.drop(columns=['PassengerId', 'Name'])\n",
        "\n",
        "\n",
        "\n"
      ],
      "metadata": {
        "id": "Nccna9tNeJLj"
      },
      "execution_count": 107,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "data.head()"
      ],
      "metadata": {
        "id": "BwfyG7HlgY49",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "outputId": "b0dff911-174a-44a3-a80c-7d47c6d15996"
      },
      "execution_count": 108,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   Survived  Pclass     Sex   Age  SibSp  Parch            Ticket     Fare  \\\n",
              "0         0       3    male  22.0      1      0         A/5 21171   7.2500   \n",
              "1         1       1  female  38.0      1      0          PC 17599  71.2833   \n",
              "2         1       3  female  26.0      0      0  STON/O2. 3101282   7.9250   \n",
              "3         1       1  female  35.0      1      0            113803  53.1000   \n",
              "4         0       3    male  35.0      0      0            373450   8.0500   \n",
              "\n",
              "  Embarked  \n",
              "0        S  \n",
              "1        C  \n",
              "2        S  \n",
              "3        S  \n",
              "4        S  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-8f70d191-a75e-48a4-8766-7b2b02a6f8c9\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Survived</th>\n",
              "      <th>Pclass</th>\n",
              "      <th>Sex</th>\n",
              "      <th>Age</th>\n",
              "      <th>SibSp</th>\n",
              "      <th>Parch</th>\n",
              "      <th>Ticket</th>\n",
              "      <th>Fare</th>\n",
              "      <th>Embarked</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>male</td>\n",
              "      <td>22.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>A/5 21171</td>\n",
              "      <td>7.2500</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>female</td>\n",
              "      <td>38.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>PC 17599</td>\n",
              "      <td>71.2833</td>\n",
              "      <td>C</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>female</td>\n",
              "      <td>26.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>STON/O2. 3101282</td>\n",
              "      <td>7.9250</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>female</td>\n",
              "      <td>35.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>113803</td>\n",
              "      <td>53.1000</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>male</td>\n",
              "      <td>35.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>373450</td>\n",
              "      <td>8.0500</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-8f70d191-a75e-48a4-8766-7b2b02a6f8c9')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-8f70d191-a75e-48a4-8766-7b2b02a6f8c9 button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-8f70d191-a75e-48a4-8766-7b2b02a6f8c9');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "<div id=\"df-eed8a204-884e-4d88-85a9-f65c57cb8bc4\">\n",
              "  <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-eed8a204-884e-4d88-85a9-f65c57cb8bc4')\"\n",
              "            title=\"Suggest charts\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "  </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "      --bg-color: #E8F0FE;\n",
              "      --fill-color: #1967D2;\n",
              "      --hover-bg-color: #E2EBFA;\n",
              "      --hover-fill-color: #174EA6;\n",
              "      --disabled-fill-color: #AAA;\n",
              "      --disabled-bg-color: #DDD;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "      --bg-color: #3B4455;\n",
              "      --fill-color: #D2E3FC;\n",
              "      --hover-bg-color: #434B5C;\n",
              "      --hover-fill-color: #FFFFFF;\n",
              "      --disabled-bg-color: #3B4455;\n",
              "      --disabled-fill-color: #666;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart {\n",
              "    background-color: var(--bg-color);\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: var(--fill-color);\n",
              "    height: 32px;\n",
              "    padding: 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: var(--hover-bg-color);\n",
              "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: var(--button-hover-fill-color);\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart-complete:disabled,\n",
              "  .colab-df-quickchart-complete:disabled:hover {\n",
              "    background-color: var(--disabled-bg-color);\n",
              "    fill: var(--disabled-fill-color);\n",
              "    box-shadow: none;\n",
              "  }\n",
              "\n",
              "  .colab-df-spinner {\n",
              "    border: 2px solid var(--fill-color);\n",
              "    border-color: transparent;\n",
              "    border-bottom-color: var(--fill-color);\n",
              "    animation:\n",
              "      spin 1s steps(1) infinite;\n",
              "  }\n",
              "\n",
              "  @keyframes spin {\n",
              "    0% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "      border-left-color: var(--fill-color);\n",
              "    }\n",
              "    20% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    30% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    40% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    60% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    80% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "    90% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "  }\n",
              "</style>\n",
              "\n",
              "  <script>\n",
              "    async function quickchart(key) {\n",
              "      const quickchartButtonEl =\n",
              "        document.querySelector('#' + key + ' button');\n",
              "      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
              "      quickchartButtonEl.classList.add('colab-df-spinner');\n",
              "      try {\n",
              "        const charts = await google.colab.kernel.invokeFunction(\n",
              "            'suggestCharts', [key], {});\n",
              "      } catch (error) {\n",
              "        console.error('Error during call to suggestCharts:', error);\n",
              "      }\n",
              "      quickchartButtonEl.classList.remove('colab-df-spinner');\n",
              "      quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
              "    }\n",
              "    (() => {\n",
              "      let quickchartButtonEl =\n",
              "        document.querySelector('#df-eed8a204-884e-4d88-85a9-f65c57cb8bc4 button');\n",
              "      quickchartButtonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "    })();\n",
              "  </script>\n",
              "</div>\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "dataframe",
              "variable_name": "data",
              "summary": "{\n  \"name\": \"data\",\n  \"rows\": 891,\n  \"fields\": [\n    {\n      \"column\": \"Survived\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 1,\n        \"num_unique_values\": 2,\n        \"samples\": [\n          1,\n          0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Pclass\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 1,\n        \"max\": 3,\n        \"num_unique_values\": 3,\n        \"samples\": [\n          3,\n          1\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Sex\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 2,\n        \"samples\": [\n          \"female\",\n          \"male\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Age\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 13.019696550973201,\n        \"min\": 0.42,\n        \"max\": 80.0,\n        \"num_unique_values\": 88,\n        \"samples\": [\n          0.75,\n          22.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"SibSp\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 1,\n        \"min\": 0,\n        \"max\": 8,\n        \"num_unique_values\": 7,\n        \"samples\": [\n          1,\n          0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Parch\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 6,\n        \"num_unique_values\": 7,\n        \"samples\": [\n          0,\n          1\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Ticket\",\n      \"properties\": {\n        \"dtype\": \"string\",\n        \"num_unique_values\": 681,\n        \"samples\": [\n          \"11774\",\n          \"248740\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Fare\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 49.6934285971809,\n        \"min\": 0.0,\n        \"max\": 512.3292,\n        \"num_unique_values\": 248,\n        \"samples\": [\n          11.2417,\n          51.8625\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Embarked\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 3,\n        \"samples\": [\n          \"S\",\n          \"C\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}"
            }
          },
          "metadata": {},
          "execution_count": 108
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "### Encode Categorical Columns"
      ],
      "metadata": {
        "id": "wCYDHp-UnNVi"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Sex and Embarked columns values are text, we can't give this text directly to the machine learning model, so we need to replace this text values to meaningful numerical values."
      ],
      "metadata": {
        "id": "j4rjNBNz0mbs"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "In Age column we will replace all male values with 0 and all the female values with 1. <br>\n",
        "and we will do the same in Embarked column: S=> 0 , C=> 1, Q => 2"
      ],
      "metadata": {
        "id": "TtSc9JYq1Wp5"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "data.replace({'Sex':{'male':0,'female':1},'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)"
      ],
      "metadata": {
        "id": "kKE48T_N2ae2"
      },
      "execution_count": 109,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "data.head()"
      ],
      "metadata": {
        "id": "SW42UKZM3u3j",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "outputId": "b1ce08df-01d6-4566-cd5f-bc89e3f4bc04"
      },
      "execution_count": 110,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   Survived  Pclass  Sex   Age  SibSp  Parch            Ticket     Fare  \\\n",
              "0         0       3    0  22.0      1      0         A/5 21171   7.2500   \n",
              "1         1       1    1  38.0      1      0          PC 17599  71.2833   \n",
              "2         1       3    1  26.0      0      0  STON/O2. 3101282   7.9250   \n",
              "3         1       1    1  35.0      1      0            113803  53.1000   \n",
              "4         0       3    0  35.0      0      0            373450   8.0500   \n",
              "\n",
              "   Embarked  \n",
              "0         0  \n",
              "1         1  \n",
              "2         0  \n",
              "3         0  \n",
              "4         0  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-e505150e-9df0-48fb-9fb9-fc1eacc06d94\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Survived</th>\n",
              "      <th>Pclass</th>\n",
              "      <th>Sex</th>\n",
              "      <th>Age</th>\n",
              "      <th>SibSp</th>\n",
              "      <th>Parch</th>\n",
              "      <th>Ticket</th>\n",
              "      <th>Fare</th>\n",
              "      <th>Embarked</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>0</td>\n",
              "      <td>22.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>A/5 21171</td>\n",
              "      <td>7.2500</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>38.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>PC 17599</td>\n",
              "      <td>71.2833</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>1</td>\n",
              "      <td>26.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>STON/O2. 3101282</td>\n",
              "      <td>7.9250</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>35.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>113803</td>\n",
              "      <td>53.1000</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>0</td>\n",
              "      <td>35.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>373450</td>\n",
              "      <td>8.0500</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-e505150e-9df0-48fb-9fb9-fc1eacc06d94')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-e505150e-9df0-48fb-9fb9-fc1eacc06d94 button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-e505150e-9df0-48fb-9fb9-fc1eacc06d94');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "<div id=\"df-41edbed1-e509-4e28-81bb-776431d77918\">\n",
              "  <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-41edbed1-e509-4e28-81bb-776431d77918')\"\n",
              "            title=\"Suggest charts\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "  </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "      --bg-color: #E8F0FE;\n",
              "      --fill-color: #1967D2;\n",
              "      --hover-bg-color: #E2EBFA;\n",
              "      --hover-fill-color: #174EA6;\n",
              "      --disabled-fill-color: #AAA;\n",
              "      --disabled-bg-color: #DDD;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "      --bg-color: #3B4455;\n",
              "      --fill-color: #D2E3FC;\n",
              "      --hover-bg-color: #434B5C;\n",
              "      --hover-fill-color: #FFFFFF;\n",
              "      --disabled-bg-color: #3B4455;\n",
              "      --disabled-fill-color: #666;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart {\n",
              "    background-color: var(--bg-color);\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: var(--fill-color);\n",
              "    height: 32px;\n",
              "    padding: 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: var(--hover-bg-color);\n",
              "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: var(--button-hover-fill-color);\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart-complete:disabled,\n",
              "  .colab-df-quickchart-complete:disabled:hover {\n",
              "    background-color: var(--disabled-bg-color);\n",
              "    fill: var(--disabled-fill-color);\n",
              "    box-shadow: none;\n",
              "  }\n",
              "\n",
              "  .colab-df-spinner {\n",
              "    border: 2px solid var(--fill-color);\n",
              "    border-color: transparent;\n",
              "    border-bottom-color: var(--fill-color);\n",
              "    animation:\n",
              "      spin 1s steps(1) infinite;\n",
              "  }\n",
              "\n",
              "  @keyframes spin {\n",
              "    0% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "      border-left-color: var(--fill-color);\n",
              "    }\n",
              "    20% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    30% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    40% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    60% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    80% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "    90% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "  }\n",
              "</style>\n",
              "\n",
              "  <script>\n",
              "    async function quickchart(key) {\n",
              "      const quickchartButtonEl =\n",
              "        document.querySelector('#' + key + ' button');\n",
              "      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
              "      quickchartButtonEl.classList.add('colab-df-spinner');\n",
              "      try {\n",
              "        const charts = await google.colab.kernel.invokeFunction(\n",
              "            'suggestCharts', [key], {});\n",
              "      } catch (error) {\n",
              "        console.error('Error during call to suggestCharts:', error);\n",
              "      }\n",
              "      quickchartButtonEl.classList.remove('colab-df-spinner');\n",
              "      quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
              "    }\n",
              "    (() => {\n",
              "      let quickchartButtonEl =\n",
              "        document.querySelector('#df-41edbed1-e509-4e28-81bb-776431d77918 button');\n",
              "      quickchartButtonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "    })();\n",
              "  </script>\n",
              "</div>\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "dataframe",
              "variable_name": "data",
              "summary": "{\n  \"name\": \"data\",\n  \"rows\": 891,\n  \"fields\": [\n    {\n      \"column\": \"Survived\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 1,\n        \"num_unique_values\": 2,\n        \"samples\": [\n          1,\n          0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Pclass\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 1,\n        \"max\": 3,\n        \"num_unique_values\": 3,\n        \"samples\": [\n          3,\n          1\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Sex\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 1,\n        \"num_unique_values\": 2,\n        \"samples\": [\n          1,\n          0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Age\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 13.019696550973201,\n        \"min\": 0.42,\n        \"max\": 80.0,\n        \"num_unique_values\": 88,\n        \"samples\": [\n          0.75,\n          22.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"SibSp\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 1,\n        \"min\": 0,\n        \"max\": 8,\n        \"num_unique_values\": 7,\n        \"samples\": [\n          1,\n          0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Parch\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 6,\n        \"num_unique_values\": 7,\n        \"samples\": [\n          0,\n          1\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Ticket\",\n      \"properties\": {\n        \"dtype\": \"string\",\n        \"num_unique_values\": 681,\n        \"samples\": [\n          \"11774\",\n          \"248740\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Fare\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 49.6934285971809,\n        \"min\": 0.0,\n        \"max\": 512.3292,\n        \"num_unique_values\": 248,\n        \"samples\": [\n          11.2417,\n          51.8625\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Embarked\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 2,\n        \"num_unique_values\": 3,\n        \"samples\": [\n          0,\n          1\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}"
            }
          },
          "metadata": {},
          "execution_count": 110
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "### Dealing with Duplicates"
      ],
      "metadata": {
        "id": "n6pu3BImnKga"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Check if there are duplicates in the dataset:"
      ],
      "metadata": {
        "id": "Uw916gfCcl6i"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "############################\n",
        "#أكمل الكود\n",
        "#Complete the code\n",
        "############################\n",
        "\n",
        "#check if there are duplicates in the dataset:\n",
        "# Check for duplicates\n",
        "duplicates = data.duplicated().sum()\n",
        "print(f'Number of duplicate rows: {duplicates}')\n",
        "\n",
        "# Optionally, print the duplicate rows\n",
        "if duplicates > 0:\n",
        "    print('Duplicate rows:')\n",
        "    print(data[data.duplicated()])\n"
      ],
      "metadata": {
        "id": "VQeshrmKnXzP",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "59978f7b-3854-44bb-d167-0229d53fcb2d"
      },
      "execution_count": 111,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Number of duplicate rows: 16\n",
            "Duplicate rows:\n",
            "     Survived  Pclass  Sex    Age  SibSp  Parch    Ticket     Fare  Embarked\n",
            "201         0       3    0  28.00      8      2  CA. 2343  69.5500         0\n",
            "324         0       3    0  28.00      8      2  CA. 2343  69.5500         0\n",
            "409         0       3    1  28.00      3      1      4133  25.4667         0\n",
            "413         0       2    0  28.00      0      0    239853   0.0000         0\n",
            "466         0       2    0  28.00      0      0    239853   0.0000         0\n",
            "485         0       3    1  28.00      3      1      4133  25.4667         0\n",
            "612         1       3    1  28.00      1      0    367230  15.5000         2\n",
            "641         1       1    1  24.00      0      0  PC 17477  69.3000         1\n",
            "644         1       3    1   0.75      2      1      2666  19.2583         1\n",
            "692         1       3    0  28.00      0      0      1601  56.4958         0\n",
            "709         1       3    0  28.00      1      1      2661  15.2458         1\n",
            "792         0       3    1  28.00      8      2  CA. 2343  69.5500         0\n",
            "826         0       3    0  28.00      0      0      1601  56.4958         0\n",
            "838         1       3    0  32.00      0      0      1601  56.4958         0\n",
            "846         0       3    0  28.00      8      2  CA. 2343  69.5500         0\n",
            "863         0       3    1  28.00      8      2  CA. 2343  69.5500         0\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "############################\n",
        "#أكمل الكود\n",
        "#Complete the code\n",
        "############################\n",
        "\n",
        "#drop the duplicates:\n",
        "\n",
        "# Check for duplicates and remove them\n",
        "data.drop_duplicates(inplace=True)\n",
        "\n"
      ],
      "metadata": {
        "id": "0YfhfO16nYza"
      },
      "execution_count": 112,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "### Data Analysis"
      ],
      "metadata": {
        "id": "q3YTIyMFta2N"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "In this section, we will explore the data and the relationships between features using statistical analysis and visualization techniques. This will help us understand the underlying patterns and correlations in the dataset, providing valuable insights for model building."
      ],
      "metadata": {
        "id": "pnaNxD4Oacgf"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "describe() provides summary statistics for numerical columns, including count, mean, standard deviation, min, max, and quartiles. This function helps us understand the distribution and central tendencies of the data. However, in our Titanic dataset, while useful, it may not be the primary focus since many insights come from categorical features and their relationships with survival, which are better explored through other means."
      ],
      "metadata": {
        "id": "y-M0sYXQbb59"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "data.describe()"
      ],
      "metadata": {
        "id": "voutv3mDamZY",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 300
        },
        "outputId": "f7d3c614-feb0-4879-f5c2-761b95d122b8"
      },
      "execution_count": 113,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "         Survived      Pclass         Sex         Age       SibSp       Parch  \\\n",
              "count  875.000000  875.000000  875.000000  875.000000  875.000000  875.000000   \n",
              "mean     0.384000    2.300571    0.350857   29.417623    0.475429    0.372571   \n",
              "std      0.486636    0.838129    0.477511   13.099904    0.947248    0.802272   \n",
              "min      0.000000    1.000000    0.000000    0.420000    0.000000    0.000000   \n",
              "25%      0.000000    2.000000    0.000000   22.000000    0.000000    0.000000   \n",
              "50%      0.000000    3.000000    0.000000   28.000000    0.000000    0.000000   \n",
              "75%      1.000000    3.000000    1.000000   35.000000    1.000000    0.000000   \n",
              "max      1.000000    3.000000    1.000000   80.000000    8.000000    6.000000   \n",
              "\n",
              "             Fare    Embarked  \n",
              "count  875.000000  875.000000  \n",
              "mean    32.007399    0.362286  \n",
              "std     49.997091    0.636563  \n",
              "min      0.000000    0.000000  \n",
              "25%      7.895800    0.000000  \n",
              "50%     14.400000    0.000000  \n",
              "75%     30.500000    1.000000  \n",
              "max    512.329200    2.000000  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-ad746848-6078-4cd2-812c-d4e1bb199643\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Survived</th>\n",
              "      <th>Pclass</th>\n",
              "      <th>Sex</th>\n",
              "      <th>Age</th>\n",
              "      <th>SibSp</th>\n",
              "      <th>Parch</th>\n",
              "      <th>Fare</th>\n",
              "      <th>Embarked</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>count</th>\n",
              "      <td>875.000000</td>\n",
              "      <td>875.000000</td>\n",
              "      <td>875.000000</td>\n",
              "      <td>875.000000</td>\n",
              "      <td>875.000000</td>\n",
              "      <td>875.000000</td>\n",
              "      <td>875.000000</td>\n",
              "      <td>875.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>mean</th>\n",
              "      <td>0.384000</td>\n",
              "      <td>2.300571</td>\n",
              "      <td>0.350857</td>\n",
              "      <td>29.417623</td>\n",
              "      <td>0.475429</td>\n",
              "      <td>0.372571</td>\n",
              "      <td>32.007399</td>\n",
              "      <td>0.362286</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>std</th>\n",
              "      <td>0.486636</td>\n",
              "      <td>0.838129</td>\n",
              "      <td>0.477511</td>\n",
              "      <td>13.099904</td>\n",
              "      <td>0.947248</td>\n",
              "      <td>0.802272</td>\n",
              "      <td>49.997091</td>\n",
              "      <td>0.636563</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>min</th>\n",
              "      <td>0.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.420000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>25%</th>\n",
              "      <td>0.000000</td>\n",
              "      <td>2.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>22.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>7.895800</td>\n",
              "      <td>0.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>50%</th>\n",
              "      <td>0.000000</td>\n",
              "      <td>3.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>28.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>14.400000</td>\n",
              "      <td>0.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>75%</th>\n",
              "      <td>1.000000</td>\n",
              "      <td>3.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>35.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>30.500000</td>\n",
              "      <td>1.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>max</th>\n",
              "      <td>1.000000</td>\n",
              "      <td>3.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>80.000000</td>\n",
              "      <td>8.000000</td>\n",
              "      <td>6.000000</td>\n",
              "      <td>512.329200</td>\n",
              "      <td>2.000000</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-ad746848-6078-4cd2-812c-d4e1bb199643')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-ad746848-6078-4cd2-812c-d4e1bb199643 button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-ad746848-6078-4cd2-812c-d4e1bb199643');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "<div id=\"df-8ae947bc-4b51-47bc-a836-daed46a824f9\">\n",
              "  <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-8ae947bc-4b51-47bc-a836-daed46a824f9')\"\n",
              "            title=\"Suggest charts\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "  </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "      --bg-color: #E8F0FE;\n",
              "      --fill-color: #1967D2;\n",
              "      --hover-bg-color: #E2EBFA;\n",
              "      --hover-fill-color: #174EA6;\n",
              "      --disabled-fill-color: #AAA;\n",
              "      --disabled-bg-color: #DDD;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "      --bg-color: #3B4455;\n",
              "      --fill-color: #D2E3FC;\n",
              "      --hover-bg-color: #434B5C;\n",
              "      --hover-fill-color: #FFFFFF;\n",
              "      --disabled-bg-color: #3B4455;\n",
              "      --disabled-fill-color: #666;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart {\n",
              "    background-color: var(--bg-color);\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: var(--fill-color);\n",
              "    height: 32px;\n",
              "    padding: 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: var(--hover-bg-color);\n",
              "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: var(--button-hover-fill-color);\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart-complete:disabled,\n",
              "  .colab-df-quickchart-complete:disabled:hover {\n",
              "    background-color: var(--disabled-bg-color);\n",
              "    fill: var(--disabled-fill-color);\n",
              "    box-shadow: none;\n",
              "  }\n",
              "\n",
              "  .colab-df-spinner {\n",
              "    border: 2px solid var(--fill-color);\n",
              "    border-color: transparent;\n",
              "    border-bottom-color: var(--fill-color);\n",
              "    animation:\n",
              "      spin 1s steps(1) infinite;\n",
              "  }\n",
              "\n",
              "  @keyframes spin {\n",
              "    0% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "      border-left-color: var(--fill-color);\n",
              "    }\n",
              "    20% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    30% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    40% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    60% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    80% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "    90% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "  }\n",
              "</style>\n",
              "\n",
              "  <script>\n",
              "    async function quickchart(key) {\n",
              "      const quickchartButtonEl =\n",
              "        document.querySelector('#' + key + ' button');\n",
              "      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
              "      quickchartButtonEl.classList.add('colab-df-spinner');\n",
              "      try {\n",
              "        const charts = await google.colab.kernel.invokeFunction(\n",
              "            'suggestCharts', [key], {});\n",
              "      } catch (error) {\n",
              "        console.error('Error during call to suggestCharts:', error);\n",
              "      }\n",
              "      quickchartButtonEl.classList.remove('colab-df-spinner');\n",
              "      quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
              "    }\n",
              "    (() => {\n",
              "      let quickchartButtonEl =\n",
              "        document.querySelector('#df-8ae947bc-4b51-47bc-a836-daed46a824f9 button');\n",
              "      quickchartButtonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "    })();\n",
              "  </script>\n",
              "</div>\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "dataframe",
              "summary": "{\n  \"name\": \"data\",\n  \"rows\": 8,\n  \"fields\": [\n    {\n      \"column\": \"Survived\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 309.21450658211774,\n        \"min\": 0.0,\n        \"max\": 875.0,\n        \"num_unique_values\": 5,\n        \"samples\": [\n          0.384,\n          1.0,\n          0.4866360501534227\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Pclass\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 308.5958133030528,\n        \"min\": 0.8381286460497531,\n        \"max\": 875.0,\n        \"num_unique_values\": 6,\n        \"samples\": [\n          875.0,\n          2.3005714285714287,\n          3.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Sex\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 309.2166417098314,\n        \"min\": 0.0,\n        \"max\": 875.0,\n        \"num_unique_values\": 5,\n        \"samples\": [\n          0.35085714285714287,\n          1.0,\n          0.47751125538492534\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Age\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 299.75250970184885,\n        \"min\": 0.42,\n        \"max\": 875.0,\n        \"num_unique_values\": 8,\n        \"samples\": [\n          29.417622857142856,\n          28.0,\n          875.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"SibSp\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 308.8444959566351,\n        \"min\": 0.0,\n        \"max\": 875.0,\n        \"num_unique_values\": 6,\n        \"samples\": [\n          875.0,\n          0.4754285714285714,\n          8.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Parch\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 309.0036351257826,\n        \"min\": 0.0,\n        \"max\": 875.0,\n        \"num_unique_values\": 5,\n        \"samples\": [\n          0.37257142857142855,\n          6.0,\n          0.8022720270339913\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Fare\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 325.84408460087917,\n        \"min\": 0.0,\n        \"max\": 875.0,\n        \"num_unique_values\": 8,\n        \"samples\": [\n          32.00739931428571,\n          14.4,\n          875.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Embarked\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 309.15799578587956,\n        \"min\": 0.0,\n        \"max\": 875.0,\n        \"num_unique_values\": 6,\n        \"samples\": [\n          875.0,\n          0.36228571428571427,\n          2.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}"
            }
          },
          "metadata": {},
          "execution_count": 113
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Look for Correlations**"
      ],
      "metadata": {
        "id": "RAb-lwTSIes-"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Now to understand the relations between the features we can use the **correlation matrix** which shows the correlation coefficients between different features in a dataset. Each cell in the matrix represents the correlation between two features. The correlation coefficient ranges from **-1 to 1**, where:<br>\n",
        "\n",
        "1 indicates a perfect positive correlation: as one feature increases, the other feature increases proportionally. <br>\n",
        "\n",
        "-1 indicates a perfect negative correlation: as one feature increases, the other feature decreases proportionally.<br>\n",
        "\n",
        "0 indicates no correlation: the features do not show any linear relationship.<br>"
      ],
      "metadata": {
        "id": "7WhEpZ2FYMTp"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "#Check for Non-Numeric Columns:\n",
        "\n",
        "non_numeric_columns = data.select_dtypes(exclude=[float, int]).columns\n",
        "print(f'Non-numeric columns: {non_numeric_columns}')\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "QXU03r1BoRkl",
        "outputId": "26134705-1714-4be8-cd94-aca2357a906e"
      },
      "execution_count": 114,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Non-numeric columns: Index(['Ticket'], dtype='object')\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#convert the result 'non numeric column' to numeric\n",
        "\n",
        "data['Ticket'] = pd.to_numeric(data['Ticket'], errors='coerce')\n"
      ],
      "metadata": {
        "id": "yuTTHZ5kocv-"
      },
      "execution_count": 115,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "data.corr()['Survived']"
      ],
      "metadata": {
        "id": "EGhTRZYwYKm4",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "50db86d6-a9a3-42e5-ee28-ce309ba4a79c"
      },
      "execution_count": 116,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Survived    1.000000\n",
              "Pclass     -0.342220\n",
              "Sex         0.552017\n",
              "Age        -0.062711\n",
              "SibSp      -0.004329\n",
              "Parch       0.093241\n",
              "Ticket     -0.121971\n",
              "Fare        0.261675\n",
              "Embarked    0.097037\n",
              "Name: Survived, dtype: float64"
            ]
          },
          "metadata": {},
          "execution_count": 116
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "The correlation values provide insights into how different features relate to the survival outcome in the Titanic dataset:\n",
        "\n",
        "* Pclass: Negative correlation (-0.338). Higher classes (lower number) are more\n",
        "likely to survive.\n",
        "* Sex: Positive correlation (0.543). Females are more likely to survive.\n",
        "* Age: Slight negative correlation (-0.070). Older passengers have a marginally lower chance of survival.\n",
        "* SibSp: Slight negative correlation (-0.035). Having more siblings/spouses aboard slightly decreases survival chances.\n",
        "* Parch: Slight positive correlation (0.082). Having more parents/children aboard slightly increases survival chances.\n",
        "* Fare: Positive correlation (0.257). Passengers who paid higher fares are more likely to survive.\n",
        "* Embarked: Slight positive correlation (0.107). The port of embarkation has a minor effect on survival.<br>\n",
        "These correlations help identify which features may be important for predicting survival."
      ],
      "metadata": {
        "id": "kM3jFWd1KEwA"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# to understand more about data lets find the number of people survived and not survived\n",
        "data['Survived'].value_counts()"
      ],
      "metadata": {
        "id": "vo9pUaoZtAdX",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "ddadcc06-7183-432b-a48e-1239155b2d10"
      },
      "execution_count": 117,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Survived\n",
              "0    539\n",
              "1    336\n",
              "Name: count, dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 117
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# making a count plot for 'Survived' column\n",
        "sns.countplot(x='Survived', data=data)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 466
        },
        "id": "ekL6bRX7thhb",
        "outputId": "1861c3e8-5c6c-438b-a8a8-4acd7779918b"
      },
      "execution_count": 118,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<Axes: xlabel='Survived', ylabel='count'>"
            ]
          },
          "metadata": {},
          "execution_count": 118
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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PzrQI0jiyg4AADAcsQMAAIxG7AAAAKMROwAAwGjEDgAAMBqxAwAAjEbsAAAAoxE7AADAaMQOAAAwGrEDAACMRuwAAACjETsAAMBoxA4AADAasQMAAIxG7AAAAKMROwAAwGjEDgAAMFqDYqd///4qLy8/b7nb7Vb//v0bOxMAAECTaVDsbNq0SdXV1ectr6qq0tatWxs9FAAAQFO5pNjZs2eP9uzZI0nav3+/7/2ePXv0ySefaNGiRfrlL3950ftbsGCBevToIavVKqvVKqfTqXXr1vnWV1VVKScnR+3bt1dkZKQyMjJUWlrqt4+SkhKlp6erbdu2io2N1aRJk3T27NlLOS0AAGCw0EvZuFevXrJYLLJYLPXeroqIiNBLL7100fu76qqrNHPmTF1//fXyer1aunSphgwZok8++UQ33nijJk6cqLVr12rlypWy2WwaN26chg0bpm3btkmSamtrlZ6eLofDoe3bt+v48eMaPXq0WrdurWefffZSTg0AABjK4vV6vRe78X//+195vV5dc8012rVrlzp27OhbFxYWptjYWLVq1apRA8XExOj555/X8OHD1bFjRy1fvlzDhw+XJB04cEDdunVTUVGRUlJStG7dOt199906duyY7Ha7JGnhwoWaPHmyTpw4obCwsIs6ptvtls1mU0VFhaxWa6Pmv5CkSa81y36By13x86MDPUKj8fMN1K+5f74v9u/vS7qy07lzZ0lSXV1d46arR21trVauXKnKyko5nU4VFxerpqZGqampvm26du2qhIQEX+wUFRWpe/fuvtCRpLS0NGVnZ2vfvn369a9/Xe+xPB6PPB6P773b7W7y8wEAAMHhkmLnxw4dOqQPP/xQZWVl58XPtGnTLno/e/fuldPpVFVVlSIjI7Vq1SolJiZq9+7dCgsLU3R0tN/2drtdLpdLkuRyufxC59z6c+suJD8/X0899dRFzwgAAC5fDYqdV199VdnZ2erQoYMcDocsFotvncViuaTYueGGG7R7925VVFTozTffVFZWljZv3tyQsS5aXl6ecnNzfe/dbrfi4+Ob9ZgAACAwGhQ7zzzzjP7yl79o8uTJjR4gLCxM1113nSQpKSlJH330kf72t7/pvvvuU3V1tcrLy/2u7pSWlsrhcEiSHA6Hdu3a5be/c7+tdW6b+oSHhys8PLzRswMAgODXoO/Z+e6773TP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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data['Sex'].value_counts()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "sBKZ4SQipL0W",
        "outputId": "b23b17cd-0a8b-4d92-b637-2476bf2a3205"
      },
      "execution_count": 119,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Sex\n",
              "0    568\n",
              "1    307\n",
              "Name: count, dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 119
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# making a count plot for 'Sex' column\n",
        "sns.countplot(x='Sex', data=data)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 466
        },
        "id": "O2meIQ0Wt9Cx",
        "outputId": "7c8443d8-14c5-47b1-fd01-74eb8b9d6e94"
      },
      "execution_count": 120,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<Axes: xlabel='Sex', ylabel='count'>"
            ]
          },
          "metadata": {},
          "execution_count": 120
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# now lets compare the number of survived beasd on the gender\n",
        "sns.countplot(x='Sex', hue='Survived', data=data)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 466
        },
        "id": "EP_gpydVuNnD",
        "outputId": "934b858b-7b07-4fa4-f1ef-7d3c7682fdb2"
      },
      "execution_count": 121,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<Axes: xlabel='Sex', ylabel='count'>"
            ]
          },
          "metadata": {},
          "execution_count": 121
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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pBA7AADAaNyzAwAAjEbsAAAAoxE7AADAaMQOAAAwGrEDAACMRuwAAACjETsAAMBoxA4AADAasQMAAIxG7AC4LLW2tqqkpESZmZmKjo6Ww+FQfn6+3nrrLatHAxBi+CJQAJelwsJCeTwebdq0Sddcc42am5tVVVWlkydPWj0agBDDmR0Al522tjbt2bNHjz76qKZMmaJhw4ZpwoQJKi8v13e+8x3fNvfee69SUlJkt9uVm5urQ4cOSfr8rJDD4dC//Mu/+Pa5d+9eRUVFqaqqypJjAtB/iB0Al534+HjFx8dr27Zt6urquuA2M2fOVEtLi3bs2KG6ujqNGzdON998sz799FOlpKRo48aN+ulPf6qDBw/q1KlTmjNnjhYsWKCbb755gI8GQH/jW88BXJZeeeUVzZ8/X5999pnGjRunb3/725o9e7bGjBmj3//+9yooKFBLS4uio6N9z7nuuuv00EMP6b777pMklZaW6j//8z81fvx4HT58WAcOHPDbHoAZiB0Al63Ozk7t2bNH+/bt044dO7R//34999xz6ujo0AMPPKDY2Fi/7T/77DP94z/+ox599FHf41GjRun48eOqq6vT6NGjrTgMAP2M2AFgjHvvvVe7du3S/fffryeffFLV1dXnbZOUlKQhQ4ZIkt577z3deOON6u7u1tatW3XnnXcO8MQABgLvxgJgjKysLG3btk3jxo2Ty+VSRESErr766gtu6/F49Pd///eaNWuWbrjhBt177706fPiwUlNTB3ZoAP2OMzsALjsnT57UzJkzdc8992jMmDFKSEjQwYMHtXDhQhUUFOi5557T5MmTderUKa1Zs0Z//dd/raamJr322mv67ne/q/Hjx2vx4sV6+eWXdejQIcXHx+vb3/62EhMTtX37dqsPD0CQETsALjtdXV366U9/qtdff10fffSRuru7lZGRoZkzZ+onP/mJYmNjderUKf3TP/2TXnnlFd9bzSdPnqxVq1bpo48+0i233KI333xT3/rWtyRJH3/8sbKzs7V69WqVlJRYfIQAgonYAQAARuNzdgAAgNGIHQAAYDRiBwAAGI3YAQAARiN2AACA0YgdAABgNGIHAAAYjdgBAABGI3YAAIDRiB0AAGA0YgcAABjt/wCq1x1f0h6EAQAAAABJRU5ErkJggg==\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "as we can see, even we have more number of male in our dataset, the number of fmale who have survived is more. this is one of the very important insight that we can get from this data."
      ],
      "metadata": {
        "id": "E7w3ebvyulnk"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# now lets compare the number of survived beasd on the Pclass\n",
        "sns.countplot(x='Pclass', hue='Survived', data=data)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 466
        },
        "id": "RRCX5YNmvorg",
        "outputId": "d00722a6-c7ec-415e-f2a1-7f73e4fab00b"
      },
      "execution_count": 122,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<Axes: xlabel='Pclass', ylabel='count'>"
            ]
          },
          "metadata": {},
          "execution_count": 122
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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jbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMLcDoAgAAuNZUzEkxugSfkDh7V4uch5EdAABgaoQdAABgaoQdAABgaoQdAABgaoQdAABgaoQdAABgaoQdAABgaoQdAABgaoQdAABgaoaGnS1btmjUqFGKj4+XxWLRmjVrPPa7XC7Nnj1bcXFxCgkJ0dChQ7V//36PNseOHVNmZqasVqsiIyM1ceJEnThxogWvAgAA+DJDw05dXZ169eqlRYsWnXf//Pnz9corr2jJkiXavn27WrdurbS0NJ06dcrdJjMzU7t379amTZu0du1abdmyRQ888EBLXQIAAPBxhn43Vnp6utLT08+7z+VyaeHChZo5c6ZGjx4tSXrrrbcUGxurNWvWaNy4cfrmm2+0fv167dixQ3369JEkvfrqqxoxYoReeOEFxcfHt9i1AAAA3+Szc3bKy8vlcDg0dOhQ97aIiAj169dPJSUlkqSSkhJFRka6g44kDR06VH5+ftq+ffsF+66vr5fT6fRYAACAOfls2HE4HJKk2NhYj+2xsbHufQ6HQzExMR77AwICFBUV5W5zPnl5eYqIiHAvCQkJXq4eAAD4Cp8NO1dSbm6uamtr3cuhQ4eMLgkAAFwhPht2bDabJKmqqspje1VVlXufzWZTdXW1x/4zZ87o2LFj7jbnExwcLKvV6rEAAABz8tmw0759e9lsNhUWFrq3OZ1Obd++XXa7XZJkt9tVU1Oj0tJSd5uPP/5YTU1N6tevX4vXDAAAfI+hT2OdOHFCBw4ccK+Xl5errKxMUVFRSkxM1LRp0/T000+rc+fOat++vWbNmqX4+HiNGTNGktStWzcNHz5ckyZN0pIlS3T69GlNmTJF48aN40ksAAAgyeCw88UXX2jQoEHu9ZycHElSVlaWli9frscee0x1dXV64IEHVFNTo1tuuUXr169Xq1at3MesWLFCU6ZM0ZAhQ+Tn56eMjAy98sorLX4tgC9KffQto0vwCaXPjze6BAAGMjTsDBw4UC6X64L7LRaL5syZozlz5lywTVRUlPLz869EeQAAwAR8ds4OAACANxB2AACAqRF2AACAqRF2AACAqRF2AACAqRF2AACAqRn66DkAtISKOSlGl+ATEmfvMroEwBCM7AAAAFMj7AAAAFMj7AAAAFMj7AAAAFMj7AAAAFPjaSwv4dulzyoIN7oCAAA8MbIDAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMzafDzpNPPimLxeKxdO3a1b3/1KlTys7OVps2bRQWFqaMjAxVVVUZWDEAAPA1Ph12JKl79+6qrKx0L59++ql73/Tp0/X+++9r9erVKi4u1uHDh3X77bcbWC0AAPA1AUYX8FMCAgJks9l+tL22tla/+93vlJ+fr8GDB0uSli1bpm7dumnbtm3q37//Bfusr69XfX29e93pdHq/cAAA4BN8fmRn//79io+PV4cOHZSZmamKigpJUmlpqU6fPq2hQ4e623bt2lWJiYkqKSm5aJ95eXmKiIhwLwkJCVf0GgAAgHF8Ouz069dPy5cv1/r167V48WKVl5frl7/8pY4fPy6Hw6GgoCBFRkZ6HBMbGyuHw3HRfnNzc1VbW+teDh06dAWvAgAAGMmnb2Olp6e7f+7Zs6f69eunpKQk/fGPf1RISEiz+w0ODlZwcLA3SgQAAD7Op0d2/llkZKSuv/56HThwQDabTQ0NDaqpqfFoU1VVdd45PgAA4Np0VYWdEydO6ODBg4qLi1NqaqoCAwNVWFjo3r9v3z5VVFTIbrcbWCUAAPAlPn0b6ze/+Y1GjRqlpKQkHT58WE888YT8/f119913KyIiQhMnTlROTo6ioqJktVo1depU2e32iz6JBQAAri0+HXa+++473X333Tp69Kiio6N1yy23aNu2bYqOjpYkLViwQH5+fsrIyFB9fb3S0tL02muvGVw1AADwJT4ddlauXHnR/a1atdKiRYu0aNGiFqoIAABcba6qOTsAAAA/F2EHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACY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flgbN2707kUCaBGEHQA+adCgQSorK9P+/fv1ww8/6M0331Tr1q0VEhJyyX3U1dUpLS1NVqtVK1as0I4dO1RQUCDp7HwcSUpPT9df//pXTZ8+XYcPH9aQIUP0m9/8RpJ04403qry8XHPnztUPP/ygu+66S3fccYf3LxbAFUXYAeCTWrdurU6dOikxMVEBAf9/x71nz54qLCy8pD727t2ro0ePat68efrlL3+prl27ekxOPic6OlpZWVn6/e9/r4ULF+r1119377NarRo7dqzeeOMNrVq1Su+8846OHTt2+RcIoMUwZwfAVSU3N1cpKSl66KGH9OCDDyooKEibN2/WnXfeqbZt23q0TUxMVFBQkF599VU9+OCD+vrrrzV37lyPNrNnz1Zqaqq6d++u+vp6rV27Vt26dZMkvfTSS4qLi9MNN9wgPz8/rV69WjabTZGRkS11uQC8gJEdAFeV66+/Xhs3btRXX32lm266SXa7Xe+++67H6M850dHRWr58uVavXq3k5GTNmzdPL7zwgkeboKAg5ebmqmfPnhowYID8/f21cuVKSVJ4eLh7QnTfvn317bff6sMPP5SfH//pBK4mFpfL5TK6CAAAgCuF/z0BAACmRtgBAACmRtgBAACmRtgBAACmRtgBAACmRtgBAACmRtgBAACmRtgBAACmRtgBAACmRtgBAACmRtgBAACm9n+iZw8WI8DY0gAAAABJRU5ErkJggg==\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "You can do the same for the other columns to get more insights about the dataset."
      ],
      "metadata": {
        "id": "OaUg5EiLvy6C"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Model Building"
      ],
      "metadata": {
        "id": "x6uCf7_YToOC"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Separating features & Target** <br><br>\n",
        "Separating features and target so that we can prepare the data for training machine learning models. In the Titanic dataset, the Survived column is the target variable, and the other columns are the features."
      ],
      "metadata": {
        "id": "RPhbd1yz39d7"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "X_train = data.drop(columns = ['Survived'], axis=1)\n",
        "y_train = data['Survived']"
      ],
      "metadata": {
        "id": "Qcy9rZ7rVbYT"
      },
      "execution_count": 123,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Splitting the data into training data & Testing data**"
      ],
      "metadata": {
        "id": "vQ3buMb94lEN"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "To build and evaluate a machine learning model effectively, it's essential to split the dataset into training and testing sets. The training set is used to train the model, allowing it to learn patterns and relationships within the data. The testing set, on the other hand, is used to evaluate the model's performance on unseen data, ensuring it can generalize well to new instances. This split helps prevent overfitting and provides a reliable estimate of the model's predictive accuracy."
      ],
      "metadata": {
        "id": "glpNpipB5x1w"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "############################\n",
        "#أكمل الكود\n",
        "#Complete the code\n",
        "############################\n",
        "from sklearn.model_selection import train_test_split\n",
        "# Split the data into training data & Testing data using train_test_split function :\n",
        "\n",
        "X_train, X_test, y_train, y_test = train_test_split(X_train,y_train, test_size=0.2, random_state=42)\n",
        "\n"
      ],
      "metadata": {
        "id": "DWbdHu7F41Vs"
      },
      "execution_count": 124,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Model Training**"
      ],
      "metadata": {
        "id": "kIeZ7-K96jdG"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Model training is a crucial step in the machine learning where the algorithm learns from the training data to make predictions. **Logistic Regression** is a commonly used algorithm for binary classification tasks, such as predicting whether a passenger survived in the Titanic dataset. By training the model on our training data, we aim to find the best-fit parameters that minimize prediction errors. Once trained, this model can be used to predict outcomes on new, unseen data."
      ],
      "metadata": {
        "id": "-RaDgM7o6i4t"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "print(\"NaNs in X_train:\", X_train.isnull().sum().sum())\n",
        "print(\"NaNs in X_test:\", X_test.isnull().sum().sum())\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "SQIuKFgzrZ1F",
        "outputId": "2cffb2f2-3798-49cd-f89e-a0ca29e0f540"
      },
      "execution_count": 125,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "NaNs in X_train: 177\n",
            "NaNs in X_test: 47\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.impute import SimpleImputer\n",
        "\n",
        "imputer = SimpleImputer(strategy='mean')\n",
        "\n",
        "# Fit the imputer on the training data and transform both training and test data\n",
        "X_train = imputer.fit_transform(X_train)\n",
        "X_test = imputer.transform(X_test)\n"
      ],
      "metadata": {
        "id": "e3UlO8WsrdS1"
      },
      "execution_count": 126,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "############################\n",
        "#أكمل الكود\n",
        "#Complete the code\n",
        "############################\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "# Create a Logistic Regression model and Train it on the training data:\n",
        "\n",
        "model = LogisticRegression()\n",
        "model.fit(X_train, y_train)\n"
      ],
      "metadata": {
        "id": "zJ5MmGR99Vsu",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 74
        },
        "outputId": "4f4a4bee-7734-48dd-f063-1d926475acb8"
      },
      "execution_count": 127,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "LogisticRegression()"
            ],
            "text/html": [
              "<style>#sk-container-id-2 {color: black;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-2\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>LogisticRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-2\" type=\"checkbox\" checked><label for=\"sk-estimator-id-2\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">LogisticRegression</label><div class=\"sk-toggleable__content\"><pre>LogisticRegression()</pre></div></div></div></div></div>"
            ]
          },
          "metadata": {},
          "execution_count": 127
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Model Evaluation"
      ],
      "metadata": {
        "id": "97DOY7s9VcOu"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Model evaluation is crucial in machine learning to assess the performance of a trained model on testing data. The **accuracy score**, a common evaluation metric, measures the proportion of correct predictions out of all predictions. This helps to gauge the model's effectiveness, ensure it generalizes well to new data, and guide further improvements."
      ],
      "metadata": {
        "id": "KkWFwbaNVgxO"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "############################\n",
        "#أكمل الكود\n",
        "#Complete the code\n",
        "############################\n",
        "from sklearn.metrics import accuracy_score\n",
        "#first let the model predict x_test\n",
        "#then use accuracy score to see the accuracy of the model\n",
        "#finally print the Accuracy.\n",
        "\n",
        "# Predict the labels for the test set\n",
        "y_pred = model.predict(X_test)\n",
        "\n",
        "# Calculate the accuracy score\n",
        "accuracy = accuracy_score(y_test, y_pred)\n",
        "print(f'Accuracy: {accuracy:.2f}')\n"
      ],
      "metadata": {
        "id": "Zs86HiXNVgBz",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "92248af4-567d-4391-8ea2-e8a0a0be5440"
      },
      "execution_count": 128,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Accuracy: 0.68\n"
          ]
        }
      ]
    }
  ]
}
