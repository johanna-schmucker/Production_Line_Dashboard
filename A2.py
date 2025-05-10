# Individual Assignment 2 Part 2

#Task Description: Assist production managers in identifying which process parameters are most associated with defective copper coils

#Imports
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#Set application page configuration
st.set_page_config(page_title = "Cable Production Decision Dashboard",layout = "wide")
st.title("Cable Production Decision Dashboard")

#Dataset Loading and Preparation
#To prevent constant re-runs of code. Functions only run once, download the data and then the result is reused
@st.cache_data
def load_and_prepare(path: str):
    df = pd.read_csv(path)
    #Label the batches by giving a defect flag (1) if any failure occurred, else 0
    #Make sure the label is an integer
    df["defect"] = ((df["Cable Failures"] > 0) | (df["Other Failures"] > 0)).astype(int)
    #Compute the total Downtime (sum of both downtime columns (cable and other failures))
    df["Total Downtime"] = (df["Cable Failure Downtime"] + df["Other Failure Downtime"])
    #Preserve the original row index for a specific batch lookup
    df["Batch ID"] = df.index
    #Parse Date if present into real pandas Timestamps
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors = "coerce")
    return df

#Call the dataset loading and preparation function
df = load_and_prepare("Cable-Production-Line-Dataset.csv")
#create a copy of the dataframe
df_original = df.copy()

#Set-Up five tabs in the application page
tab1, tab2, tab3, tab4 = st.tabs(["1. Overview", "2. Machine Analysis", "3. Shift & Operator", "4. Recommendations"])


#Tab 1: Overview of the Dataset and specific batches
with tab1:
    #Tab page title
    st.header("1. Data Overview")
    st.subheader("Sample Production Data")
    #Display the first 6 rows of the Dataset
    st.dataframe(df.head(6))

    #Get the key overall metrics for this production sample
    st.subheader("Key Metrics")
    total_batches = len(df)
    #Calculate the overall defect rate in percentages
    overall_defect_rate = df["defect"].mean() * 100
    #Calculate the average downtme for all defects
    average_downtime = df["Total Downtime"].mean()
    #Creates three side-by-side column containers
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Batches", total_batches)
    c2.metric("Overall Defect Rate", f"{overall_defect_rate: .1f}%")
    c3.metric("Average Downtime/Batch", f"{average_downtime: .1f} min")

    #Visulaization: Overlaid histograms to get a first impression on three main factors influencing operations
    st.subheader("Downtime by Machine / Shift / Operators")
    figure, axes = plt.subplots(1, 3, figsize = (18, 4), sharey = True)
    #Histogram 1: by Machine for all 17 machines
    sns.histplot(df, x = "Total Downtime", hue = "Machine", element = "step", stat = "count", common_norm = False, ax = axes[0])
    axes[0].set_title("By Machine")
    #Histogram 2: by Shift for both shifts
    sns.histplot(df, x = "Total Downtime", hue = "Shift", element = "step", stat = "count", common_norm = False, ax = axes[1])
    axes[1].set_title("By Shift")
    #Histogram 3: top 5 Operators
    #Get the data from the top 3 operators
    operators = df["Operator"].value_counts().index
    sns.histplot(df[df["Operator"].isin(operators)], x = "Total Downtime", hue = "Operator", element = "step", stat = "count", common_norm = False, ax = axes[2])
    axes[2].set_title("By Operators")
    #Plotting of the diagrams will be donw via a for loop
    for x in axes:
        x.set_xlabel("Downtime (min)")
        x.set_ylabel("Batch Count")
        x.legend(loc = "upper right", fontsize = 8)
    #Plotting
    st.pyplot(figure)

    #Inspect a specific batch of the sample
    st.subheader("Inspect a specific Batch")
    #Check the batch the user specifies via a user input
    batch_input = st.text_input(f"Enter Batch ID (0–{total_batches-1}): ")
    if batch_input:
        #Checks if the user actually inputted something. If it is empty the rest of this block will be skipped
        try:
            #Convert the input to an integer
            bid = int(batch_input)
            #Assigns the valid input to the according batch id column
            batch_row = df_original[df_original["Batch ID"] == bid]
            #If the integer cannot be matched, a warning will be issued
            if batch_row.empty:
                st.warning("Batch ID not found.")
            #Otherwise display that single-row DataFrame so the user sees all its columns
            else:
                st.dataframe(batch_row)
                #Display batch metrics (key numbers for that batch)
                downtime = batch_row["Total Downtime"].iloc[0]
                cable_failures = batch_row["Cable Failures"].iloc[0]
                other_failures = batch_row["Other Failures"].iloc[0]
                is_defective = batch_row["defect"].iloc[0]
                #Assign a human-readable label (defective or not defective) to the batch
                if is_defective == 1:
                    flag = "Yes"
                else:
                    flag = "No"
                #Creates four side-by-side column containers for displayal
                b1, b2, b3, b4 = st.columns(4)
                b1.metric("Downtime (min)", f"{downtime:.1f}")
                b2.metric("Cable Failures", cable_failures)
                b3.metric("Other Failures", other_failures)
                b4.metric("Defective Batch?", flag)


                #Visualization: Batch vs Overall Downtime Histogramm Plots
                st.subheader("Batch vs Overall Downtime")
                figure, axes = plt.subplots()
                sns.histplot(df["Total Downtime"], bins = 30, kde = True, color = "skyblue", ax = axes)
                axes.axvline(downtime, color = "red", linestyle = "--", label = f"Batch {bid}")
                axes.set_xlabel("Total Downtime (min)")
                axes.set_ylabel("Batch Count")
                axes.legend()
                #Plotting
                st.pyplot(figure)

                # Batch vs Overall Failures
                st.subheader("Batch vs Overall Failures")
                figure, axes = plt.subplots(1, 2, figsize = (12, 4))
                axes_1, axes_2 = axes
                #Subplot 1
                sns.boxplot(y = df["Cable Failures"], ax = axes_1, color = "skyblue")
                axes_1.scatter(0, cable_failures, color = "red", zorder = 10)
                axes_1.set_title("Cable Failures")
                axes_1.set_ylabel("Count")
                #Subplot 2
                sns.boxplot(y = df["Other Failures"], ax = axes_2, color = "skyblue")
                axes_2.scatter(0, other_failures, color = "red", zorder = 10)
                axes_2.set_title("Other Failures")
                axes_2.set_ylabel("Count")
                #Plotting
                st.pyplot(figure)
        #Displays a warning telling the user that the input must be an integer
        except ValueError:
            st.warning("Please enter a valid integer.")

#Tab 2: Machine Level Analysis
with tab2:
    st.header("2. Machine-Level Analysis")
    #Aggregate metrics. Group the data by the machines
    #Count how many batches each machine processed and sum up how many of those were defective
    #And then compute average downtime and average failures per batch
    mstats = df.groupby("Machine").agg(Batch_Count = ("Batch ID", "count"), Total_Defects = ("defect", "sum"), Average_Downtime = ("Total Downtime", "mean"), Average_CableFails = ("Cable Failures", "mean"), Average_OtherFails = ("Other Failures", "mean")).reset_index()
    mstats["Defect Rate (%)"] = (mstats["Total_Defects"] / mstats["Batch_Count"] * 100)
    st.subheader("Shift Performance Summary")
    st.dataframe(mstats)

    #Display key metrics
    num_machines = df["Machine"].nunique()
    avg_machine_defect = df.groupby("Machine")["defect"].mean().mean() * 100
    worst_machine = (df.groupby("Machine")["Total Downtime"].mean().idxmax())
    km1, km2, km3 = st.columns(3)
    km1.metric("Total Machines", num_machines)
    km2.metric("Avg Defect Rate per Machine", f"{avg_machine_defect:.1f}%")
    km3.metric("Machine with Max Avg Downtime", worst_machine)

    #Visualization: combined graph - Dual-axis Downtime vs total Failures
    st.subheader("Downtime vs Total Failures")
    figure, axes_1 = plt.subplots(figsize = (8,5))
    sns.barplot(data = mstats, x = "Machine", y = "Average_Downtime", color = "lightgray", ax = axes_1)
    axes_1.set_ylabel("Average Downtime (min)")
    axes_2 = axes_1.twinx()
    axes_2.plot(mstats["Machine"], mstats["Average_CableFails"], marker = "o", label = "Cable Fails")
    axes_2.plot(mstats["Machine"], mstats["Average_OtherFails"], marker = "s", label = "Other Fails")
    axes_2.set_ylabel("Average Failures per Batch")
    axes_1.set_title("Machine Downtime & Failures")
    axes_1.grid(axis = "y", linestyle = "--", alpha = 0.3)
    axes_2.legend(loc = "upper right")
    #Plotting
    st.pyplot(figure)

    #Visualization: Boxplot of the total Downtime by Machine
    st.subheader("Total Downtime Boxplot by Machine")
    figure, axes = plt.subplots(figsize = (10,5))
    sns.boxplot(data = df, x = "Machine", y = "Total Downtime", palette = "pastel", ax = axes)
    axes.set_ylabel("Downtime (min)")
    axes.set_title("Downtime Variability per Machine")
    axes.grid(axis = "y", linestyle = "--", alpha = 0.3)
    #Plotting
    st.pyplot(figure)


#Tab 3: Shift and Operator Analysis
with tab3:
    st.header("3. Shift & Operator Analysis")
    #Aggregate metrics. Group the data by shifts
    #Count how many batches each machine processed and sum up how many of those were defective
    #And then compute average downtime and average failures per batch
    mstats = df.groupby("Shift").agg(Batch_Count = ("Batch ID", "count"), Total_Defects  = ("defect", "sum"), Average_Downtime = ("Total Downtime", "mean"), Average_CableFails = ("Cable Failures", "mean"), Average_OtherFails = ("Other Failures", "mean")).reset_index()
    mstats["Defect Rate (%)"] = (mstats["Total_Defects"] / mstats["Batch_Count"] * 100)

    st.subheader("Shift Performance Summary")
    st.dataframe(mstats)
    #Display key metrics
    num_shifts = df["Shift"].nunique()
    num_operators = df["Operator"].nunique()
    avg_op_defect = df.groupby("Operator")["defect"].mean().mean() * 100
    ks1, ks2, ks3 = st.columns(3)
    ks1.metric("Total Shifts", num_shifts)
    ks2.metric("Total Operators", num_operators)
    ks3.metric("Avg Operator Defect Rate", f"{avg_op_defect:.1f}%")


    #Groupby Operators
    grouped = df.groupby("Operator")
    #Aggregate the two metrics (Experience and Defects)
    operator_statistics = grouped.agg(Experience = ("Batch ID", "count"), Defect_Rate0 = ("defect", "mean"))
    #Convert the 0–1 mean into a percentage
    operator_statistics["Defect_Rate"] = operator_statistics["Defect_Rate0"] * 100
    #Drop the temporary column and turn index
    operator_statistics = (operator_statistics.drop(columns = "Defect_Rate0").reset_index())

    #Visualization: Operator Downtime vs Total Defects per Shift
    st.subheader("Operator Downtime vs Total Defects per Shift")
    #Aggregate average downtime and total defects
    #Create the GroupBy object
    grouped_shift_operator = df.groupby(["Shift", "Operator"])
    #Aggregate into a new DataFrame
    shift_operator_df = grouped_shift_operator.agg(Average_Downtime = ("Total Downtime", "mean"), Total_Defects = ("defect", "sum"))
    #Turn the grouping keys back into columns
    shift_operator_df = shift_operator_df.reset_index()

    #Get all the unique Shifts
    shifts = shift_operator_df["Shift"].unique()

    #Visualization: combined Bar Plot for Operators, Shifts and the Average Downtime
    figure, axes = plt.subplots(1, len(shifts), figsize = (18, 5), sharey = False)
    #Plotting of the diagrams will be donw via a for loop
    for i, shift in enumerate(shifts):
        subplot = shift_operator_df[shift_operator_df["Shift"] == shift]
        #Convert operator IDs to strings for the x-axis
        operators = subplot["Operator"].astype(str).tolist()
        downtime = subplot["Average_Downtime"].tolist()
        fails = subplot["Total_Defects"].tolist()
        axes_1 = axes[i]
        #Plot downtime bars
        axes_1.bar(operators, downtime, color = "skyblue")
        axes_1.set_title(f"Shift {shift}")
        axes_1.set_xlabel("Operator")
        axes_1.set_ylabel("Average Downtime (min)")
        axes_1.tick_params(axis = "x", rotation = 45)

        #Plot total defects line
        axes_2 = axes_1.twinx()
        axes_2.plot(operators, fails, marker = "o", color = "C3", label = "Total Defects")
        axes_2.set_ylabel("Total Defects")
        axes_2.legend(loc = "upper right")

    #Combine both plots into a tight one
    figure.tight_layout()
    #Plotting
    st.pyplot(figure)

    #Operator Experience vs Defect Rate
    st.subheader("Operator Experience vs Defect Rate")
    #Visualization: Regression Plot about the correlation between the Experience and and detected Defects
    figure, axes = plt.subplots(figsize = (8, 5))
    sns.regplot(data = operator_statistics, x = "Experience", y = "Defect_Rate", scatter_kws = {"s": 60, "alpha": 0.7, "color": "steelblue"}, line_kws = {"color": "red", "linewidth": 2}, ax = axes)
    axes.set_xlabel("Experience (Batches Processed)")
    axes.set_ylabel("Defect Rate (%)")
    axes.set_title("Experience vs Defect Rate")
    axes.grid(True, linestyle = "--", alpha = 0.3)
    #Plotting
    st.pyplot(figure)


#Tab 4: Conclusion
with tab4:
    st.header("4. Recommendations to Managers")
    #Prepare label encoders which one will use to store each LabelEncoder instance by column name
    label_dictionary = {}
    df_encoder = df.copy()
    #Loops over the three categorical columns one wants to encode
    for column in ["Machine","Shift","Operator"]:
        #Object will learn to map each unique category in the column to a unique integer
        label_encoder = LabelEncoder()
        #Ensures all values in column x are strings
        df_encoder[column] = label_encoder.fit_transform(df_encoder[column].astype(str))
        #Stores the fitted LabelEncoder instance in your dictionary under the key column
        label_dictionary[column] = label_encoder

    #Feature Matrix
    X = df_encoder[["Machine","Shift","Operator"]]
    #Label Vector
    y = df_encoder["defect"]
    #Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.20, random_state = 42)
    #Used Machine Learning Model with 100 seperate trees
    model = RandomForestClassifier(n_estimators = 100, random_state = 42)
    model.fit(X_train, y_train)
    #Compute accuracy
    predictions = model.predict(X_test)
    accurracy = accuracy_score(y_test, predictions)
    #Compute downtime‐savings
    total_dt = df["Total Downtime"].sum()
    per_machine_dt = df.groupby("Machine")["Total Downtime"].sum()
    worst_machine = per_machine_dt.idxmax()
    batches_worst = len(df[df["Machine"]== worst_machine])
    average_dt = total_dt / len(df)
    saved = per_machine_dt[worst_machine] - batches_worst * average_dt

    #Key metrics
    df_encoder["Pred_Prob"] = model.predict_proba(X)[:,1] 
    #Compute confidence of top combo:
    combo_probs = (df_encoder.groupby(["Machine","Shift","Operator"])["Pred_Prob"].mean().reset_index(name = "Defect_Prob").sort_values("Defect_Prob"))
    best_defect_prob = combo_probs["Defect_Prob"].iloc[0]
    recommendation_conf = (1 - best_defect_prob) * 100
 
    #Calculation for the optimal Feature Combination
    st.subheader("Feature Importance")
    #Creates a dataframe which computes the feature importances for all parameter combinations
    feature_importance = pd.DataFrame({"Feature": ["Machine","Shift","Operator"],"Importance": model.feature_importances_}).sort_values("Importance", ascending = False)
    figure, axes = plt.subplots(figsize = (8,4))
    #Visualize the Feature Importance Plot
    sns.barplot(x = "Importance", y = "Feature", data = feature_importance, ax = axes)
    st.pyplot(figure)

    #Top three non-defective parameter combinations by lowest defect-probability
    st.subheader("Top 3 Parameter Combos by Lowest Predicted Defect Risk")
    top_3 = combo_probs.head(3).copy()
    #Decode labels back to original
    #Loops over all categorical fields one encoded before
    #Best_combinations is a small Series where each entry is the integer‐encoded label that appears most frequently among defect-free predictions
    for column in ["Machine","Shift","Operator"]:
        top_3[column] = label_dictionary[column].inverse_transform(top_3[column])
    #Add a Confidence column
    top_3["Confidence (%)"] = (1 - top_3["Defect_Prob"]) * 100
    st.dataframe(top_3[["Machine","Shift","Operator","Confidence (%)"]])

    #Action Plan narrative
    st.subheader("Action Plan")
    #Creates two side-by-side column containers
    c1, c2 = st.columns(2)
    c1.metric("Total Downtime (min)", int(total_dt))
    c2.metric("Improvement if Replace", f"{saved:.0f} min")
    #Display the final recommendation
    st.markdown(
        f"**Recommendation:** Replace or optimize Machine **{worst_machine}** "
        f"to save ~**{saved:.0f}** minutes of downtime.")
