from ydata_profiling import ProfileReport

def generate_profile_report(df):

    profile = ProfileReport(
        df,
        title="Automated EDA Report",
        explorative=True
    )

    return profile