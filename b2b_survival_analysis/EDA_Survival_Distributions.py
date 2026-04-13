import pandas as pd
import matplotlib.pyplot as plt
from sksurv.nonparametric import kaplan_meier_estimator

def load_data():
    return pd.read_csv('static_data.csv')

def plot_global_survival(df):
    time, survival_prob = kaplan_meier_estimator(df["event_observed"].astype(bool), df["time_to_event"])
    plt.step(time, survival_prob, where="post")
    plt.title("Global Kaplan-Meier Survival Curve")
    plt.ylabel("Survival Probability")
    plt.xlabel("Time (Months)")
    plt.ylim([0, 1.05])
    plt.savefig('global_survival.png')
    plt.clf()

def plot_segment_survival(df):
    for segment in df["account_segment"].unique():
        mask = df["account_segment"] == segment
        time, survival_prob = kaplan_meier_estimator(
            df["event_observed"][mask].astype(bool), 
            df["time_to_event"][mask])
        plt.step(time, survival_prob, where="post", label=f"{segment}")

    plt.legend()
    plt.title("Survival by Account Segment")
    plt.ylabel("Probability of active subscription")
    plt.xlabel("Contract Months")
    plt.savefig('segment_survival.png')
    plt.clf()

if __name__ == "__main__":
    df = load_data()
    plot_global_survival(df)
    plot_segment_survival(df)
    print("Survival curves saved successfully.")
