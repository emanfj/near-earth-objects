{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "29b965a6",
   "metadata": {
    "_cell_guid": "b1076dfc-b9ad-4769-8c92-a6c4dae69d19",
    "_uuid": "8f2839f25d086af736a60e9eeb907d3b93b6e0e5",
    "execution": {
     "iopub.execute_input": "2024-08-07T17:56:38.238473Z",
     "iopub.status.busy": "2024-08-07T17:56:38.238065Z",
     "iopub.status.idle": "2024-08-07T17:56:53.128035Z",
     "shell.execute_reply": "2024-08-07T17:56:53.126877Z"
    },
    "papermill": {
     "duration": 14.898962,
     "end_time": "2024-08-07T17:56:53.130599",
     "exception": false,
     "start_time": "2024-08-07T17:56:38.231637",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2024-08-07 17:56:41.006134: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:9261] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered\n",
      "2024-08-07 17:56:41.006266: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:607] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered\n",
      "2024-08-07 17:56:41.155351: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1515] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import tensorflow as tf\n",
    "import matplotlib.pyplot as plt\n",
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import Dense, Input\n",
    "from tensorflow.keras.activations import relu\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import LabelEncoder, StandardScaler\n",
    "from sklearn.metrics import confusion_matrix, classification_report"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "b855a45e",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-08-07T17:56:53.142150Z",
     "iopub.status.busy": "2024-08-07T17:56:53.141473Z",
     "iopub.status.idle": "2024-08-07T17:56:54.009265Z",
     "shell.execute_reply": "2024-08-07T17:56:54.008059Z"
    },
    "papermill": {
     "duration": 0.876392,
     "end_time": "2024-08-07T17:56:54.011863",
     "exception": false,
     "start_time": "2024-08-07T17:56:53.135471",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "data=pd.read_csv('/kaggle/input/nasa-nearest-earth-objects-1910-2024/nearest-earth-objects(1910-2024).csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "9850efa1",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-08-07T17:56:54.023061Z",
     "iopub.status.busy": "2024-08-07T17:56:54.022634Z",
     "iopub.status.idle": "2024-08-07T17:56:54.089398Z",
     "shell.execute_reply": "2024-08-07T17:56:54.087891Z"
    },
    "papermill": {
     "duration": 0.075104,
     "end_time": "2024-08-07T17:56:54.091833",
     "exception": false,
     "start_time": "2024-08-07T17:56:54.016729",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 338199 entries, 0 to 338198\n",
      "Data columns (total 9 columns):\n",
      " #   Column                  Non-Null Count   Dtype  \n",
      "---  ------                  --------------   -----  \n",
      " 0   neo_id                  338199 non-null  int64  \n",
      " 1   name                    338199 non-null  object \n",
      " 2   absolute_magnitude      338171 non-null  float64\n",
      " 3   estimated_diameter_min  338171 non-null  float64\n",
      " 4   estimated_diameter_max  338171 non-null  float64\n",
      " 5   orbiting_body           338199 non-null  object \n",
      " 6   relative_velocity       338199 non-null  float64\n",
      " 7   miss_distance           338199 non-null  float64\n",
      " 8   is_hazardous            338199 non-null  bool   \n",
      "dtypes: bool(1), float64(5), int64(1), object(2)\n",
      "memory usage: 21.0+ MB\n"
     ]
    }
   ],
   "source": [
    "data.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "d82f27ad",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-08-07T17:56:54.103183Z",
     "iopub.status.busy": "2024-08-07T17:56:54.102755Z",
     "iopub.status.idle": "2024-08-07T17:56:54.148577Z",
     "shell.execute_reply": "2024-08-07T17:56:54.147494Z"
    },
    "papermill": {
     "duration": 0.054203,
     "end_time": "2024-08-07T17:56:54.150852",
     "exception": false,
     "start_time": "2024-08-07T17:56:54.096649",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "neo_id                     0\n",
       "name                       0\n",
       "absolute_magnitude        28\n",
       "estimated_diameter_min    28\n",
       "estimated_diameter_max    28\n",
       "orbiting_body              0\n",
       "relative_velocity          0\n",
       "miss_distance              0\n",
       "is_hazardous               0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "1596de2b",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-08-07T17:56:54.162327Z",
     "iopub.status.busy": "2024-08-07T17:56:54.161910Z",
     "iopub.status.idle": "2024-08-07T17:56:54.243088Z",
     "shell.execute_reply": "2024-08-07T17:56:54.241834Z"
    },
    "papermill": {
     "duration": 0.0909,
     "end_time": "2024-08-07T17:56:54.246746",
     "exception": false,
     "start_time": "2024-08-07T17:56:54.155846",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
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
       "      <th>neo_id</th>\n",
       "      <th>name</th>\n",
       "      <th>absolute_magnitude</th>\n",
       "      <th>estimated_diameter_min</th>\n",
       "      <th>estimated_diameter_max</th>\n",
       "      <th>orbiting_body</th>\n",
       "      <th>relative_velocity</th>\n",
       "      <th>miss_distance</th>\n",
       "      <th>is_hazardous</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2162117</td>\n",
       "      <td>162117 (1998 SD15)</td>\n",
       "      <td>19.140</td>\n",
       "      <td>0.394962</td>\n",
       "      <td>0.883161</td>\n",
       "      <td>Earth</td>\n",
       "      <td>71745.401048</td>\n",
       "      <td>5.814362e+07</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2349507</td>\n",
       "      <td>349507 (2008 QY)</td>\n",
       "      <td>18.500</td>\n",
       "      <td>0.530341</td>\n",
       "      <td>1.185878</td>\n",
       "      <td>Earth</td>\n",
       "      <td>109949.757148</td>\n",
       "      <td>5.580105e+07</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2455415</td>\n",
       "      <td>455415 (2003 GA)</td>\n",
       "      <td>21.450</td>\n",
       "      <td>0.136319</td>\n",
       "      <td>0.304818</td>\n",
       "      <td>Earth</td>\n",
       "      <td>24865.506798</td>\n",
       "      <td>6.720689e+07</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>3132126</td>\n",
       "      <td>(2002 PB)</td>\n",
       "      <td>20.630</td>\n",
       "      <td>0.198863</td>\n",
       "      <td>0.444672</td>\n",
       "      <td>Earth</td>\n",
       "      <td>78890.076805</td>\n",
       "      <td>3.039644e+07</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>3557844</td>\n",
       "      <td>(2011 DW)</td>\n",
       "      <td>22.700</td>\n",
       "      <td>0.076658</td>\n",
       "      <td>0.171412</td>\n",
       "      <td>Earth</td>\n",
       "      <td>56036.519484</td>\n",
       "      <td>6.311863e+07</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>338194</th>\n",
       "      <td>54403809</td>\n",
       "      <td>(2023 VS4)</td>\n",
       "      <td>28.580</td>\n",
       "      <td>0.005112</td>\n",
       "      <td>0.011430</td>\n",
       "      <td>Earth</td>\n",
       "      <td>56646.985988</td>\n",
       "      <td>6.406548e+07</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>338195</th>\n",
       "      <td>54415298</td>\n",
       "      <td>(2023 XW5)</td>\n",
       "      <td>28.690</td>\n",
       "      <td>0.004859</td>\n",
       "      <td>0.010865</td>\n",
       "      <td>Earth</td>\n",
       "      <td>21130.768947</td>\n",
       "      <td>2.948883e+07</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>338196</th>\n",
       "      <td>54454871</td>\n",
       "      <td>(2024 KJ7)</td>\n",
       "      <td>21.919</td>\n",
       "      <td>0.109839</td>\n",
       "      <td>0.245607</td>\n",
       "      <td>Earth</td>\n",
       "      <td>11832.041031</td>\n",
       "      <td>5.346078e+07</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>338197</th>\n",
       "      <td>54456245</td>\n",
       "      <td>(2024 NE)</td>\n",
       "      <td>23.887</td>\n",
       "      <td>0.044377</td>\n",
       "      <td>0.099229</td>\n",
       "      <td>Earth</td>\n",
       "      <td>56198.382733</td>\n",
       "      <td>5.184742e+06</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>338198</th>\n",
       "      <td>54460573</td>\n",
       "      <td>(2024 NH3)</td>\n",
       "      <td>22.951</td>\n",
       "      <td>0.068290</td>\n",
       "      <td>0.152700</td>\n",
       "      <td>Earth</td>\n",
       "      <td>42060.357830</td>\n",
       "      <td>7.126682e+06</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>338171 rows × 9 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "          neo_id                name  absolute_magnitude  \\\n",
       "0        2162117  162117 (1998 SD15)              19.140   \n",
       "1        2349507    349507 (2008 QY)              18.500   \n",
       "2        2455415    455415 (2003 GA)              21.450   \n",
       "3        3132126           (2002 PB)              20.630   \n",
       "4        3557844           (2011 DW)              22.700   \n",
       "...          ...                 ...                 ...   \n",
       "338194  54403809          (2023 VS4)              28.580   \n",
       "338195  54415298          (2023 XW5)              28.690   \n",
       "338196  54454871          (2024 KJ7)              21.919   \n",
       "338197  54456245           (2024 NE)              23.887   \n",
       "338198  54460573          (2024 NH3)              22.951   \n",
       "\n",
       "        estimated_diameter_min  estimated_diameter_max orbiting_body  \\\n",
       "0                     0.394962                0.883161         Earth   \n",
       "1                     0.530341                1.185878         Earth   \n",
       "2                     0.136319                0.304818         Earth   \n",
       "3                     0.198863                0.444672         Earth   \n",
       "4                     0.076658                0.171412         Earth   \n",
       "...                        ...                     ...           ...   \n",
       "338194                0.005112                0.011430         Earth   \n",
       "338195                0.004859                0.010865         Earth   \n",
       "338196                0.109839                0.245607         Earth   \n",
       "338197                0.044377                0.099229         Earth   \n",
       "338198                0.068290                0.152700         Earth   \n",
       "\n",
       "        relative_velocity  miss_distance  is_hazardous  \n",
       "0            71745.401048   5.814362e+07         False  \n",
       "1           109949.757148   5.580105e+07          True  \n",
       "2            24865.506798   6.720689e+07         False  \n",
       "3            78890.076805   3.039644e+07         False  \n",
       "4            56036.519484   6.311863e+07         False  \n",
       "...                   ...            ...           ...  \n",
       "338194       56646.985988   6.406548e+07         False  \n",
       "338195       21130.768947   2.948883e+07         False  \n",
       "338196       11832.041031   5.346078e+07         False  \n",
       "338197       56198.382733   5.184742e+06         False  \n",
       "338198       42060.357830   7.126682e+06         False  \n",
       "\n",
       "[338171 rows x 9 columns]"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.dropna()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "442f5d58",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-08-07T17:56:54.267174Z",
     "iopub.status.busy": "2024-08-07T17:56:54.266763Z",
     "iopub.status.idle": "2024-08-07T17:56:54.394927Z",
     "shell.execute_reply": "2024-08-07T17:56:54.393809Z"
    },
    "papermill": {
     "duration": 0.137293,
     "end_time": "2024-08-07T17:56:54.397245",
     "exception": false,
     "start_time": "2024-08-07T17:56:54.259952",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
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
       "      <th>neo_id</th>\n",
       "      <th>absolute_magnitude</th>\n",
       "      <th>estimated_diameter_min</th>\n",
       "      <th>estimated_diameter_max</th>\n",
       "      <th>relative_velocity</th>\n",
       "      <th>miss_distance</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>3.381990e+05</td>\n",
       "      <td>338171.000000</td>\n",
       "      <td>338171.000000</td>\n",
       "      <td>338171.000000</td>\n",
       "      <td>338199.000000</td>\n",
       "      <td>3.381990e+05</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>1.759939e+07</td>\n",
       "      <td>22.932525</td>\n",
       "      <td>0.157812</td>\n",
       "      <td>0.352878</td>\n",
       "      <td>51060.662908</td>\n",
       "      <td>4.153535e+07</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>2.287225e+07</td>\n",
       "      <td>2.911216</td>\n",
       "      <td>0.313885</td>\n",
       "      <td>0.701869</td>\n",
       "      <td>26399.238435</td>\n",
       "      <td>2.077399e+07</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>2.000433e+06</td>\n",
       "      <td>9.250000</td>\n",
       "      <td>0.000511</td>\n",
       "      <td>0.001143</td>\n",
       "      <td>203.346433</td>\n",
       "      <td>6.745533e+03</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>3.373980e+06</td>\n",
       "      <td>20.740000</td>\n",
       "      <td>0.025384</td>\n",
       "      <td>0.056760</td>\n",
       "      <td>30712.031471</td>\n",
       "      <td>2.494540e+07</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>3.742127e+06</td>\n",
       "      <td>22.800000</td>\n",
       "      <td>0.073207</td>\n",
       "      <td>0.163697</td>\n",
       "      <td>47560.465474</td>\n",
       "      <td>4.332674e+07</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>5.405374e+07</td>\n",
       "      <td>25.100000</td>\n",
       "      <td>0.189041</td>\n",
       "      <td>0.422708</td>\n",
       "      <td>66673.820614</td>\n",
       "      <td>5.933961e+07</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>5.446281e+07</td>\n",
       "      <td>33.580000</td>\n",
       "      <td>37.545248</td>\n",
       "      <td>83.953727</td>\n",
       "      <td>291781.106613</td>\n",
       "      <td>7.479865e+07</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "             neo_id  absolute_magnitude  estimated_diameter_min  \\\n",
       "count  3.381990e+05       338171.000000           338171.000000   \n",
       "mean   1.759939e+07           22.932525                0.157812   \n",
       "std    2.287225e+07            2.911216                0.313885   \n",
       "min    2.000433e+06            9.250000                0.000511   \n",
       "25%    3.373980e+06           20.740000                0.025384   \n",
       "50%    3.742127e+06           22.800000                0.073207   \n",
       "75%    5.405374e+07           25.100000                0.189041   \n",
       "max    5.446281e+07           33.580000               37.545248   \n",
       "\n",
       "       estimated_diameter_max  relative_velocity  miss_distance  \n",
       "count           338171.000000      338199.000000   3.381990e+05  \n",
       "mean                 0.352878       51060.662908   4.153535e+07  \n",
       "std                  0.701869       26399.238435   2.077399e+07  \n",
       "min                  0.001143         203.346433   6.745533e+03  \n",
       "25%                  0.056760       30712.031471   2.494540e+07  \n",
       "50%                  0.163697       47560.465474   4.332674e+07  \n",
       "75%                  0.422708       66673.820614   5.933961e+07  \n",
       "max                 83.953727      291781.106613   7.479865e+07  "
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "dfd51285",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-08-07T17:56:54.409845Z",
     "iopub.status.busy": "2024-08-07T17:56:54.409455Z",
     "iopub.status.idle": "2024-08-07T17:56:54.417369Z",
     "shell.execute_reply": "2024-08-07T17:56:54.416311Z"
    },
    "papermill": {
     "duration": 0.016859,
     "end_time": "2024-08-07T17:56:54.419616",
     "exception": false,
     "start_time": "2024-08-07T17:56:54.402757",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "X = data[['absolute_magnitude', 'estimated_diameter_min', 'estimated_diameter_max', 'relative_velocity', 'miss_distance']]\n",
    "y = data['is_hazardous']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "2910c018",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-08-07T17:56:54.432694Z",
     "iopub.status.busy": "2024-08-07T17:56:54.431973Z",
     "iopub.status.idle": "2024-08-07T17:56:54.450732Z",
     "shell.execute_reply": "2024-08-07T17:56:54.449669Z"
    },
    "papermill": {
     "duration": 0.028168,
     "end_time": "2024-08-07T17:56:54.453314",
     "exception": false,
     "start_time": "2024-08-07T17:56:54.425146",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "encoder = LabelEncoder()\n",
    "y = encoder.fit_transform(y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "e9d7d21b",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-08-07T17:56:54.466048Z",
     "iopub.status.busy": "2024-08-07T17:56:54.465610Z",
     "iopub.status.idle": "2024-08-07T17:56:54.522422Z",
     "shell.execute_reply": "2024-08-07T17:56:54.521238Z"
    },
    "papermill": {
     "duration": 0.066174,
     "end_time": "2024-08-07T17:56:54.525038",
     "exception": false,
     "start_time": "2024-08-07T17:56:54.458864",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "scaler = StandardScaler()\n",
    "X=scaler.fit_transform(X)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "547efa72",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-08-07T17:56:54.538174Z",
     "iopub.status.busy": "2024-08-07T17:56:54.537753Z",
     "iopub.status.idle": "2024-08-07T17:56:54.582320Z",
     "shell.execute_reply": "2024-08-07T17:56:54.581375Z"
    },
    "papermill": {
     "duration": 0.053881,
     "end_time": "2024-08-07T17:56:54.584792",
     "exception": false,
     "start_time": "2024-08-07T17:56:54.530911",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "535f7749",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-08-07T17:56:54.597834Z",
     "iopub.status.busy": "2024-08-07T17:56:54.597442Z",
     "iopub.status.idle": "2024-08-07T17:56:54.690775Z",
     "shell.execute_reply": "2024-08-07T17:56:54.689831Z"
    },
    "papermill": {
     "duration": 0.102685,
     "end_time": "2024-08-07T17:56:54.693319",
     "exception": false,
     "start_time": "2024-08-07T17:56:54.590634",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "model = Sequential([\n",
    "    Input(shape=(X_train.shape[1],)),\n",
    "    Dense(units=5, activation='relu'),\n",
    "    Dense(units=3, activation='relu'),\n",
    "    Dense(units=1, activation='sigmoid')\n",
    "])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "ea4e020d",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-08-07T17:56:54.705791Z",
     "iopub.status.busy": "2024-08-07T17:56:54.705396Z",
     "iopub.status.idle": "2024-08-07T17:56:54.722118Z",
     "shell.execute_reply": "2024-08-07T17:56:54.721050Z"
    },
    "papermill": {
     "duration": 0.025847,
     "end_time": "2024-08-07T17:56:54.724643",
     "exception": false,
     "start_time": "2024-08-07T17:56:54.698796",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "4bddcff4",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-08-07T17:56:54.738261Z",
     "iopub.status.busy": "2024-08-07T17:56:54.737264Z",
     "iopub.status.idle": "2024-08-07T18:05:35.764508Z",
     "shell.execute_reply": "2024-08-07T18:05:35.763493Z"
    },
    "papermill": {
     "duration": 521.036771,
     "end_time": "2024-08-07T18:05:35.767107",
     "exception": false,
     "start_time": "2024-08-07T17:56:54.730336",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/50\n",
      "\u001b[1m6764/6764\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m11s\u001b[0m 2ms/step - accuracy: 0.8727 - loss: nan - val_accuracy: 0.8729 - val_loss: nan\n",
      "Epoch 2/50\n",
      "\u001b[1m6764/6764\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m10s\u001b[0m 1ms/step - accuracy: 0.8719 - loss: nan - val_accuracy: 0.8729 - val_loss: nan\n",
      "Epoch 3/50\n",
      "\u001b[1m6764/6764\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m10s\u001b[0m 1ms/step - accuracy: 0.8718 - loss: nan - val_accuracy: 0.8729 - val_loss: nan\n",
      "Epoch 4/50\n",
      "\u001b[1m6764/6764\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m10s\u001b[0m 2ms/step - accuracy: 0.8720 - loss: nan - val_accuracy: 0.8729 - val_loss: nan\n",
      "Epoch 5/50\n",
      "\u001b[1m6764/6764\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m10s\u001b[0m 1ms/step - accuracy: 0.8721 - loss: nan - val_accuracy: 0.8729 - val_loss: nan\n",
      "Epoch 6/50\n",
      "\u001b[1m6764/6764\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m10s\u001b[0m 1ms/step - accuracy: 0.8718 - loss: nan - val_accuracy: 0.8729 - val_loss: nan\n",
      "Epoch 7/50\n",
      "\u001b[1m6764/6764\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m10s\u001b[0m 2ms/step - accuracy: 0.8717 - loss: nan - val_accuracy: 0.8729 - val_loss: nan\n",
      "Epoch 8/50\n",
      "\u001b[1m6764/6764\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m10s\u001b[0m 1ms/step - accuracy: 0.8728 - loss: nan - val_accuracy: 0.8729 - val_loss: nan\n",
      "Epoch 9/50\n",
      "\u001b[1m6764/6764\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m10s\u001b[0m 1ms/step - accuracy: 0.8724 - loss: nan - val_accuracy: 0.8729 - val_loss: nan\n",
      "Epoch 10/50\n",
      "\u001b[1m6764/6764\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m10s\u001b[0m 1ms/step - accuracy: 0.8725 - loss: nan - val_accuracy: 0.8729 - val_loss: nan\n",
      "Epoch 11/50\n",
      "\u001b[1m6764/6764\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m10s\u001b[0m 1ms/step - accuracy: 0.8711 - loss: nan - val_accuracy: 0.8729 - val_loss: nan\n",
      "Epoch 12/50\n",
      "\u001b[1m6764/6764\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m10s\u001b[0m 2ms/step - accuracy: 0.8711 - loss: nan - val_accuracy: 0.8729 - val_loss: nan\n",
      "Epoch 13/50\n",
      "\u001b[1m6764/6764\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m10s\u001b[0m 1ms/step - accuracy: 0.8734 - loss: nan - val_accuracy: 0.8729 - val_loss: nan\n",
      "Epoch 14/50\n",
      "\u001b[1m6764/6764\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m10s\u001b[0m 2ms/step - accuracy: 0.8733 - loss: nan - val_accuracy: 0.8729 - val_loss: nan\n",
      "Epoch 15/50\n",
      "\u001b[1m6764/6764\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m10s\u001b[0m 1ms/step - accuracy: 0.8734 - loss: nan - val_accuracy: 0.8729 - val_loss: nan\n",
      "Epoch 16/50\n",
      "\u001b[1m6764/6764\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m10s\u001b[0m 1ms/step - accuracy: 0.8714 - loss: nan - val_accuracy: 0.8729 - val_loss: nan\n",
      "Epoch 17/50\n",
      "\u001b[1m6764/6764\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m10s\u001b[0m 1ms/step - accuracy: 0.8720 - loss: nan - val_accuracy: 0.8729 - val_loss: nan\n",
      "Epoch 18/50\n",
      "\u001b[1m6764/6764\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m10s\u001b[0m 1ms/step - accuracy: 0.8721 - loss: nan - val_accuracy: 0.8729 - val_loss: nan\n",
      "Epoch 19/50\n",
      "\u001b[1m6764/6764\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m10s\u001b[0m 2ms/step - accuracy: 0.8718 - loss: nan - val_accuracy: 0.8729 - val_loss: nan\n",
      "Epoch 20/50\n",
      "\u001b[1m6764/6764\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m10s\u001b[0m 2ms/step - accuracy: 0.8726 - loss: nan - val_accuracy: 0.8729 - val_loss: nan\n",
      "Epoch 21/50\n",
      "\u001b[1m6764/6764\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m10s\u001b[0m 2ms/step - accuracy: 0.8712 - loss: nan - val_accuracy: 0.8729 - val_loss: nan\n",
      "Epoch 22/50\n",
      "\u001b[1m6764/6764\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m10s\u001b[0m 2ms/step - accuracy: 0.8719 - loss: nan - val_accuracy: 0.8729 - val_loss: nan\n",
      "Epoch 23/50\n",
      "\u001b[1m6764/6764\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m11s\u001b[0m 2ms/step - accuracy: 0.8715 - loss: nan - val_accuracy: 0.8729 - val_loss: nan\n",
      "Epoch 24/50\n",
      "\u001b[1m6764/6764\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m10s\u001b[0m 2ms/step - accuracy: 0.8712 - loss: nan - val_accuracy: 0.8729 - val_loss: nan\n",
      "Epoch 25/50\n",
      "\u001b[1m6764/6764\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m10s\u001b[0m 2ms/step - accuracy: 0.8721 - loss: nan - val_accuracy: 0.8729 - val_loss: nan\n",
      "Epoch 26/50\n",
      "\u001b[1m6764/6764\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m11s\u001b[0m 2ms/step - accuracy: 0.8725 - loss: nan - val_accuracy: 0.8729 - val_loss: nan\n",
      "Epoch 27/50\n",
      "\u001b[1m6764/6764\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m10s\u001b[0m 2ms/step - accuracy: 0.8719 - loss: nan - val_accuracy: 0.8729 - val_loss: nan\n",
      "Epoch 28/50\n",
      "\u001b[1m6764/6764\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m10s\u001b[0m 2ms/step - accuracy: 0.8725 - loss: nan - val_accuracy: 0.8729 - val_loss: nan\n",
      "Epoch 29/50\n",
      "\u001b[1m6764/6764\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m11s\u001b[0m 2ms/step - accuracy: 0.8741 - loss: nan - val_accuracy: 0.8729 - val_loss: nan\n",
      "Epoch 30/50\n",
      "\u001b[1m6764/6764\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m11s\u001b[0m 2ms/step - accuracy: 0.8724 - loss: nan - val_accuracy: 0.8729 - val_loss: nan\n",
      "Epoch 31/50\n",
      "\u001b[1m6764/6764\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m10s\u001b[0m 2ms/step - accuracy: 0.8712 - loss: nan - val_accuracy: 0.8729 - val_loss: nan\n",
      "Epoch 32/50\n",
      "\u001b[1m6764/6764\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m11s\u001b[0m 2ms/step - accuracy: 0.8716 - loss: nan - val_accuracy: 0.8729 - val_loss: nan\n",
      "Epoch 33/50\n",
      "\u001b[1m6764/6764\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m10s\u001b[0m 2ms/step - accuracy: 0.8730 - loss: nan - val_accuracy: 0.8729 - val_loss: nan\n",
      "Epoch 34/50\n",
      "\u001b[1m6764/6764\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m10s\u001b[0m 2ms/step - accuracy: 0.8725 - loss: nan - val_accuracy: 0.8729 - val_loss: nan\n",
      "Epoch 35/50\n",
      "\u001b[1m6764/6764\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m10s\u001b[0m 2ms/step - accuracy: 0.8729 - loss: nan - val_accuracy: 0.8729 - val_loss: nan\n",
      "Epoch 36/50\n",
      "\u001b[1m6764/6764\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m10s\u001b[0m 2ms/step - accuracy: 0.8725 - loss: nan - val_accuracy: 0.8729 - val_loss: nan\n",
      "Epoch 37/50\n",
      "\u001b[1m6764/6764\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m10s\u001b[0m 2ms/step - accuracy: 0.8705 - loss: nan - val_accuracy: 0.8729 - val_loss: nan\n",
      "Epoch 38/50\n",
      "\u001b[1m6764/6764\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m11s\u001b[0m 2ms/step - accuracy: 0.8731 - loss: nan - val_accuracy: 0.8729 - val_loss: nan\n",
      "Epoch 39/50\n",
      "\u001b[1m6764/6764\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m10s\u001b[0m 2ms/step - accuracy: 0.8716 - loss: nan - val_accuracy: 0.8729 - val_loss: nan\n",
      "Epoch 40/50\n",
      "\u001b[1m6764/6764\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m10s\u001b[0m 2ms/step - accuracy: 0.8714 - loss: nan - val_accuracy: 0.8729 - val_loss: nan\n",
      "Epoch 41/50\n",
      "\u001b[1m6764/6764\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m10s\u001b[0m 2ms/step - accuracy: 0.8725 - loss: nan - val_accuracy: 0.8729 - val_loss: nan\n",
      "Epoch 42/50\n",
      "\u001b[1m6764/6764\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m11s\u001b[0m 2ms/step - accuracy: 0.8731 - loss: nan - val_accuracy: 0.8729 - val_loss: nan\n",
      "Epoch 43/50\n",
      "\u001b[1m6764/6764\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m11s\u001b[0m 2ms/step - accuracy: 0.8723 - loss: nan - val_accuracy: 0.8729 - val_loss: nan\n",
      "Epoch 44/50\n",
      "\u001b[1m6764/6764\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m11s\u001b[0m 2ms/step - accuracy: 0.8725 - loss: nan - val_accuracy: 0.8729 - val_loss: nan\n",
      "Epoch 45/50\n",
      "\u001b[1m6764/6764\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m11s\u001b[0m 2ms/step - accuracy: 0.8720 - loss: nan - val_accuracy: 0.8729 - val_loss: nan\n",
      "Epoch 46/50\n",
      "\u001b[1m6764/6764\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m11s\u001b[0m 2ms/step - accuracy: 0.8726 - loss: nan - val_accuracy: 0.8729 - val_loss: nan\n",
      "Epoch 47/50\n",
      "\u001b[1m6764/6764\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m11s\u001b[0m 2ms/step - accuracy: 0.8715 - loss: nan - val_accuracy: 0.8729 - val_loss: nan\n",
      "Epoch 48/50\n",
      "\u001b[1m6764/6764\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m11s\u001b[0m 2ms/step - accuracy: 0.8729 - loss: nan - val_accuracy: 0.8729 - val_loss: nan\n",
      "Epoch 49/50\n",
      "\u001b[1m6764/6764\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m11s\u001b[0m 2ms/step - accuracy: 0.8718 - loss: nan - val_accuracy: 0.8729 - val_loss: nan\n",
      "Epoch 50/50\n",
      "\u001b[1m6764/6764\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m11s\u001b[0m 2ms/step - accuracy: 0.8722 - loss: nan - val_accuracy: 0.8729 - val_loss: nan\n"
     ]
    }
   ],
   "source": [
    "history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "df296344",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-08-07T18:05:37.022458Z",
     "iopub.status.busy": "2024-08-07T18:05:37.021904Z",
     "iopub.status.idle": "2024-08-07T18:05:39.365446Z",
     "shell.execute_reply": "2024-08-07T18:05:39.364060Z"
    },
    "papermill": {
     "duration": 3.002611,
     "end_time": "2024-08-07T18:05:39.367835",
     "exception": false,
     "start_time": "2024-08-07T18:05:36.365224",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Test accuracy: 0.8724\n"
     ]
    }
   ],
   "source": [
    "loss, accuracy = model.evaluate(X_test, y_test, verbose=0)\n",
    "print(f\"Test accuracy: {accuracy:.4f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "fd0b888b",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-08-07T18:05:40.543556Z",
     "iopub.status.busy": "2024-08-07T18:05:40.543144Z",
     "iopub.status.idle": "2024-08-07T18:05:43.715967Z",
     "shell.execute_reply": "2024-08-07T18:05:43.714790Z"
    },
    "papermill": {
     "duration": 3.769445,
     "end_time": "2024-08-07T18:05:43.718698",
     "exception": false,
     "start_time": "2024-08-07T18:05:39.949253",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m2114/2114\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m2s\u001b[0m 1ms/step\n"
     ]
    }
   ],
   "source": [
    "predictions = model.predict(X_test)\n",
    "predictions = (predictions > 0.5).astype(int)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "311d78f5",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-08-07T18:05:44.981602Z",
     "iopub.status.busy": "2024-08-07T18:05:44.981150Z",
     "iopub.status.idle": "2024-08-07T18:05:45.620350Z",
     "shell.execute_reply": "2024-08-07T18:05:45.619175Z"
    },
    "papermill": {
     "duration": 1.250483,
     "end_time": "2024-08-07T18:05:45.622675",
     "exception": false,
     "start_time": "2024-08-07T18:05:44.372192",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAABKUAAAGGCAYAAACqvTJ0AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuNSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/xnp5ZAAAACXBIWXMAAA9hAAAPYQGoP6dpAACJh0lEQVR4nOzdd1gUV/s38O/SQZqFIgbFQhQMlqCgmMSGghJiL1gARXlMAAuaKArWKCYaQwyWFMQYRAm2GDUqYFeiRKKxYEdQERQJrIIC7s77h6/zc2VBUNhV/H6ua67HPXPmnHtOfLKTe885IxEEQQAREREREREREZEKaag7ACIiIiIiIiIievswKUVERERERERERCrHpBQREREREREREakck1JERERERERERKRyTEoREREREREREZHKMSlFREREREREREQqx6QUERERERERERGpHJNSRERERERERESkckxKERERERERERGRyjEpRURvPIlEgrlz51b5uuvXr0MikWDt2rXVHhMRERFRbcdnMCJ6VUxKEVG1WLt2LSQSCSQSCY4cOVLmvCAIsLa2hkQiwccff6yGCKvHrl27IJFIYGVlBblcru5wiIiI6C1Xm5/BDhw4AIlEgk2bNqk7FCKqIUxKEVG10tPTQ2xsbJnygwcP4ubNm9DV1VVDVNVn/fr1sLGxwe3bt7Fv3z51h0NEREQEoPY/gxFR7cSkFBFVq759+yI+Ph6PHz9WKI+NjYWjoyMsLS3VFNmrKywsxO+//47g4GC0b98e69evV3dI5SosLFR3CERERKRCtfkZjIhqLyaliKhaeXl54d69e0hISBDLSkpKsGnTJowYMULpNYWFhZg6dSqsra2hq6uLli1bYunSpRAEQaFecXExpkyZAjMzMxgZGeGTTz7BzZs3lbZ569YtjB07FhYWFtDV1UXr1q2xZs2aV7q3rVu34uHDhxgyZAiGDx+OLVu24NGjR2XqPXr0CHPnzsW7774LPT09NGzYEAMHDsTVq1fFOnK5HN999x0cHBygp6cHMzMzuLu74++//wZQ8V4Lz+/fMHfuXEgkEpw/fx4jRoxA3bp18cEHHwAA/v33X/j6+qJZs2bQ09ODpaUlxo4di3v37ikdMz8/P1hZWUFXVxdNmzbFp59+ipKSEly7dg0SiQTffvttmeuOHTsGiUSCDRs2VHVIiYiIqJrU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in0PHjxYoY3Hjx8LS5YsEVq1aiXo6OgIZmZmQp8+fYSTJ0+KdYqKigQ/Pz/BxMREMDIyEoYOHSrcuXOnzP3OmTNHACDcvXu3TGw3b94UBgwYIJiamgomJibCkCFDhKysLKVjlpGRIXh7ewtmZmaCrq6u0KxZMyEgIEAoLi4u027r1q0FDQ0N4ebNm+WOCxEREdW82voMJgiCsH//fgFAucfhw4cFQRCEq1evCoMHDxZMTU0FPT09wcnJSdixY4dCWz/88IPw0UcfCfXr1xd0dXWF5s2bC59//rlQUFAgCIIgFBcXC59//rnQtm1bwcjISKhTp47Qtm1bYeXKlRXGSESvRiIIz63rICIieoH27dujXr16SEpKUncoRERERET0huKeUkREVCV///03Tp06BW9vb3WHQkREREREbzDOlCIioko5e/YsTp48iW+++Qa5ubm4du2a+KYaIiIiIiKiquJMKSIiqpRNmzZhzJgxKC0txYYNG5iQIiIiIiKiV8KZUkREREREREREpHKcKUVERERERERERCrHpBQREREREREREamclroDqM3kcjmysrJgZGQEiUSi7nCIiIjoFQmCgPv378PKygoaGvxtr7rwmYmIiKh2qewzE5NSNSgrKwvW1tbqDoOIiIiq2Y0bN/DOO++oO4xag89MREREtdOLnpmYlKpBRkZGAJ78QzA2NlZzNERERPSqpFIprK2txe94qh58ZiIiIqpdKvvMxKRUDXo6/dzY2JgPWERERLUIl5hVLz4zERER1U4vembiZghERERERERERKRyTEoREREREREREZHKMSlFREREREREREQqxz2liIiIiIiIiGopuVyOkpISdYdBtYy2tjY0NTVfuR0mpYiIiIiIiIhqoZKSEqSnp0Mul6s7FKqFTE1NYWlp+UovgGFSioiIiIiIiKiWEQQBt2/fhqamJqytraGhwd17qHoIgoCioiLcuXMHANCwYcOXbotJKSIiIiIiIqJa5vHjxygqKoKVlRUMDAzUHQ7VMvr6+gCAO3fuwNzc/KWX8jFVSkRERERERFTLyGQyAICOjo6aI6Ha6mmys7S09KXbYFKKiIiIiIiIqJZ6lf1+iCpSHX+3uHzvTSQIQGmRuqMgIiJ6vWkbAHwQJyIiInptMSn1JiotAhZZqTsKIiKi19vMLECnjrqjICIiIjWzsbHB5MmT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xAQFBQV82CIiIqoFasN3e1xcHHx8fPDDDz/AyckJERER+O2333DhwgVYWFjA29sbjRo1Qnh4OIAnSxi6du2KxYsXw8PDAxs3bsSiRYuQmpqK9957T2kfNjY2mDx5stJXbCtTG8aViOh18+jRI6Snp6Np06bQ09NTdzhUC1X0d6yy3+3c6JyIiIjoLTJs2DDcvXsXs2fPRnZ2Ntq1a4fdu3eLywgyMzOhofF/k+ldXFwQGxuL0NBQzJw5E7a2tti2bVu5CSkiIiKiyuJMqRrEX/2IiIhqF3631wyOKxFR9eNMKapp1TFTintKERERERERERGRyjEpRUREREREREREKsekFBERERERERERqRyTUkRERERERES1FLeRppoil8tfuQ2+fY+IiIiIiIioltHW1oZEIsHdu3dhZmYGiUSi7pColhAEASUlJbh79y40NDSgo6Pz0m0xKUVERERERERUy2hqauKdd97BzZs3cf36dXWHQ7WQgYEBGjduDA2Nl1+Ex6QUERERERERUS1kaGgIW1tblJaWqjsUqmU0NTWhpaX1yjPwmJQiIiIiIiIiqqU0NTWhqamp7jCIlOJG50REREREREREpHJMShERERERERERkcoxKUVERERERERERCrHpBQREREREREREakck1JERERERERERKRyak9KrVixAjY2NtDT04OzszNOnDhRYf2IiAi0bNkS+vr6sLa2xpQpU/Do0SPxvI2NDSQSSZkjICAAAJCXl4egoCCxjcaNG2PixIkoKChQ6CclJQU9e/aEqakp6tatCzc3N5w+fbr6B4CIiIiIiIiI6C2k1qRUXFwcgoODMWfOHKSmpqJt27Zwc3PDnTt3lNaPjY3FjBkzMGfOHKSlpSEqKgpxcXGYOXOmWCclJQW3b98Wj4SEBADAkCFDAABZWVnIysrC0qVLcfbsWaxduxa7d++Gn5+f2MaDBw/g7u6Oxo0b4/jx4zhy5AiMjIzg5uaG0tLSGhwRIiIiIiIiIqK3g0QQBEFdnTs7O6Njx46IjIwEAMjlclhbWyMoKAgzZswoUz8wMBBpaWlISkoSy6ZOnSomjpSZPHkyduzYgcuXL0MikSitEx8fj1GjRqGwsBBaWlr4+++/0bFjR2RmZsLa2hoAcObMGbRp0waXL19GixYtKnV/UqkUJiYmKCgogLGxcaWuISIiotcXv9tr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kyfjyy+/rFR8y5YtQ926deHi4gJPT0+4ubnh/fffV6izatUqDB48GJ999hlatWqF8ePHK8wmA5788y8pKcGYMWOqOkREREREVEMkwrMbPFRCSUkJRo8ejfj4eGhpPdmSSi6Xw9vbG6tXr4aOjk6NBPomkkqlMDExQUFBAYyNjdUdDhHRK3n06BHS09PRtGlT6OnpqTscoio5fPgwevbsiRs3brxwVllFf9er47t94MCBFZ7Pz8/HwYMHX7icszbhMxMREVHtUtnv9ipvdK6jo4O4uDh8+eWXOHXqFPT19eHg4IAmTZq8UsBERERE1a24uBh3797F3LlzMWTIkJde5lidTExMXnje29tbRdEQERERqU+Vk1JP2drawtbWtjpjISIiIqpWGzZsgJ+fH9q1a4d169apOxwAQHR0tLpDICIiInotVHlPqUGDBuGrr74qU/71119jyJAh1RIUERERUXXw9fWFTCbDyZMn0ahRI3WHQ0RERETPqHJS6tChQ+jbt2+Z8j59+uDQoUPVEhQREREREREREdVuVU5KPXjwQOlm5tra2pBKpdUSFBERERERERER1W5VTko5ODggLi6uTPnGjRthb29fLUEREdHrq4ovbSV64/DvOBEREZFqVHmj87CwMAwcOBBXr15Fjx49AABJSUmIjY3Fpk2bqj1AIiJ6PWhrawMAioqKoK+vr+ZoiGpOUVERgP/7O09ERERENaPKSSlPT09s27YNixYtwqZNm6Cvr4+2bdti3759qFevXk3ESERErwFNTU2Ymprizp07AAADAwNIJBI1R0VUfQRBQFFREe7cuQNTU1NoamqqOyQiIiKiWq3KSSkA8PDwgIeHBwBAKpViw4YNmDZtGk6ePAmZTFatARIR0evD0tISAMTEFFFtZGpqKv5dJyIiIqKa81JJKeDJW/iioqKwefNmWFlZYeDAgVixYkV1xkZERK8ZiUSChg0bwtzcHKWlpeoOh6jaaWtrc4YUERERkYpUKSmVnZ2NtWvXIioqClKpFEOHDkVxcTG2bdv2Spucr1ixAkuWLEF2djbatm2L77//Hk5OTuXWj4iIwKpVq5CZmYkGDRpg8ODBCA8Ph56eHgDAxsYGGRkZZa777LPPsGLFCuTl5WHOnDnYu3cvMjMzYWZmhv79+2PBggUwMTER6ytblrJhwwYMHz78pe+ViKg20NTU5H+4ExERERHRK6l0UsrT0xOHDh2Ch4cHIiIi4O7uDk1NTaxevfqVAoiLi0NwcDBWr14NZ2dnREREwM3NDRcvXoS5uXmZ+rGxsZgxYwbWrFkDFxcXXLp0Cb6+vpBIJFi2bBkAICUlRWEZ4dmzZ9GrVy8MGTIEAJCVlYWsrCwsXboU9vb2yMjIwIQJE5CVlVVms/bo6Gi4u7uLn01NTV/pfomIiIiIiIiIqApJqT///BMTJ07Ep59+Cltb22oLYNmyZRg/fjzGjBkDAFi9ejV27tyJNWvWYMaMGWXqHzt2DF26dMGIESMAPJkV5eXlhePHj4t1zMzMFK5ZvHgxmjdvjq5duwIA3nvvPWzevFk837x5cyxcuBCjRo3C48ePoaX1f8PCfSWIiIiIiIiIiKqfRmUrHjlyBPfv34ejoyOcnZ0RGRmJ3NzcV+q8pKQEJ0+ehKur6/8FpKEBV1dXJCcnK73GxcUFJ0+exIkTJwAA165dw65du9C3b99y+4iJicHYsWMrfEtUQUEBjI2NFRJSABAQEIAGDRrAyckJa9asgSAI5bZRXFwMqVSqcBARERERERERUVmVTkp16tQJP/30E27fvo3//e9/2LhxI6ysrCCXy5GQkID79+9XufPc3FzIZDJYWFgolFtYWCA7O1vpNSNGjMD8+fPxwQcfQFtbG82bN0e3bt0wc+ZMpfW3bduG/Px8+Pr6VhjHggUL4O/vr1A+f/58/Pbbb0hISMCgQYPw2Wef4fvvvy+3nfDwcJiYmIiHtbV1uXWJiIiIiIiIiN5mEqGiqT8vcPHiRURFReHXX39Ffn4+evXqhe3bt1f6+qysLDRq1AjHjh1D586dxfIvvvgCBw8eVFiS99SBAwcwfPhwfPnll3B2dsaVK1cwadIkjB8/HmFhYWXqu7m5QUdHB3/88YfSGKRSKXr16oV69eph+/bt0NbWLjfe2bNnIzo6Gjdu3FB6vri4GMXFxQptW1tbi7OwiIiI6M0mlUphYmLC7/ZqxnElIiKqXSr73V7pmVLKtGzZEl9//TVu3ryJDRs2VPn6Bg0aQFNTEzk5OQrlOTk55e7jFBYWhtGjR2PcuHFwcHDAgAEDsGjRIoSHh0MulyvUzcjIQGJiIsaNG6e0rfv378Pd3R1GRkbYunVrhQkpAHB2dsbNmzcVEk/P0tXVhbGxscJBRERERERERERlvVJS6ilNTU3079+/SrOkAEBHRweOjo5ISkoSy+RyOZKSkhRmTj2rqKgIGhqKYT99Lfnzk76io6Nhbm4ODw+PMu1IpVL07t0bOjo62L59O/T09F4Y76lTp1C3bl3o6uq+sC4REREREREREZWv0m/fqynBwcHw8fFBhw4d4OTkhIiICBQWFopv4/P29kajRo0QHh4OAPD09MSyZcvQvn17cfleWFgYPD09xeQU8CS5FR0dDR8fnzKblz9NSBUVFSEmJkZhU3IzMzNoamrijz/+QE5ODjp16gQ9PT0kJCRg0aJFmDZtmopGhoiIiIiIiIio9lJ7UmrYsGG4e/cuZs+ejezsbLRr1w67d+8WNz/PzMxUmBkVGhoKiUSC0NBQ3Lp1C2ZmZvD09MTChQsV2k1MTERmZibGjh1bps/U1FRxv6oWLVoonEtPT4eNjQ20tbWxYsUKTJkyBYIgoEWLFli2bBnGjx9f3UNARERERERERPTWeaWNzqli3LSTiIioduF3e83guBIREdUuKtnonIiIiIiIiIiI6GUwKUVERERERERERCrHpBQREREREREREakck1JERERERERERKRyTEoREREREREREZHKMSlFREREREREREQqx6QUERERERERERGpHJNSRERERERERESkckxKERERERERERGRyjEpRUREREREREREKsekFBERERERERERqRyTUkREREREREREpHJMShERERERERERkcoxKUVERERERERERCrHpBQRERHRWyIvLw8jR46EsbExTE1N4efnhwcPHlR4zaNHjxAQEID69evD0NAQgwYNQk5Ojnj+9OnT8PLygrW1NfT19WFnZ4fvvvuupm+FiIiIagEmpYiIiIjeEiNHjsS5c+eQkJCAHTt24NChQ/D396/wmilTpuCPP/5AfHw8Dh48iKysLAwcOFA8f/LkSZibmyMmJgbnzp3DrFmzEBISgsjIyJq+HSIiInrDSQRBENQdRG0llUphYmKCgoICGBsbqzscIiIiekVv8nd7Wloa7O3tkZKSgg4dOgAAdu/ejb59++LmzZuwsrIqc01BQQHMzMwQGxuLwYMHAwAuXLgAOzs7JCcno1OnTkr7CggIQFpaGvbt21ep2N7kcSUiIqKyKvvdzplSRERERG+B5ORkmJqaigkpAHB1dYWGhgaOHz+u9JqTJ0+itLQUrq6u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",
      "text/plain": [
       "<Figure size 1200x400 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(12, 4))\n",
    "plt.subplot(1, 2, 1)\n",
    "plt.plot(history.history['accuracy'], label='Training Accuracy')\n",
    "plt.plot(history.history['val_accuracy'], label='Validation Accuracy')\n",
    "plt.title('Model Accuracy')\n",
    "plt.xlabel('Epoch')\n",
    "plt.ylabel('Accuracy')\n",
    "plt.legend()\n",
    "\n",
    "plt.subplot(1, 2, 2)\n",
    "plt.plot(history.history['loss'], label='Training Loss')\n",
    "plt.plot(history.history['val_loss'], label='Validation Loss')\n",
    "plt.title('Model Loss')\n",
    "plt.xlabel('Epoch')\n",
    "plt.ylabel('Loss')\n",
    "plt.legend()\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "284461e3",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-08-07T18:05:46.870950Z",
     "iopub.status.busy": "2024-08-07T18:05:46.870044Z",
     "iopub.status.idle": "2024-08-07T18:05:46.997189Z",
     "shell.execute_reply": "2024-08-07T18:05:46.995746Z"
    },
    "papermill": {
     "duration": 0.718253,
     "end_time": "2024-08-07T18:05:46.999813",
     "exception": false,
     "start_time": "2024-08-07T18:05:46.281560",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Confusion Matrix:\n",
      "[[59011     0]\n",
      " [ 8629     0]]\n",
      "\n",
      "Classification Report:\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.87      1.00      0.93     59011\n",
      "           1       0.00      0.00      0.00      8629\n",
      "\n",
      "    accuracy                           0.87     67640\n",
      "   macro avg       0.44      0.50      0.47     67640\n",
      "weighted avg       0.76      0.87      0.81     67640\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/opt/conda/lib/python3.10/site-packages/sklearn/metrics/_classification.py:1344: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.\n",
      "  _warn_prf(average, modifier, msg_start, len(result))\n",
      "/opt/conda/lib/python3.10/site-packages/sklearn/metrics/_classification.py:1344: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.\n",
      "  _warn_prf(average, modifier, msg_start, len(result))\n",
      "/opt/conda/lib/python3.10/site-packages/sklearn/metrics/_classification.py:1344: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.\n",
      "  _warn_prf(average, modifier, msg_start, len(result))\n"
     ]
    }
   ],
   "source": [
    "cm = confusion_matrix(y_test, predictions)\n",
    "print(\"Confusion Matrix:\")\n",
    "print(cm)\n",
    "\n",
    "print(\"\\nClassification Report:\")\n",
    "print(classification_report(y_test, predictions))"
   ]
  }
 ],
 "metadata": {
  "kaggle": {
   "accelerator": "none",
   "dataSources": [
    {
     "datasetId": 5410371,
     "sourceId": 8984243,
     "sourceType": "datasetVersion"
    }
   ],
   "dockerImageVersionId": 30746,
   "isGpuEnabled": false,
   "isInternetEnabled": false,
   "language": "python",
   "sourceType": "notebook"
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.13"
  },
  "papermill": {
   "default_parameters": {},
   "duration": 554.448141,
   "end_time": "2024-08-07T18:05:49.949927",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2024-08-07T17:56:35.501786",
   "version": "2.5.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
