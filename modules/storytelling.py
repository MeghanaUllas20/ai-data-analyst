def generate_story(df):
    rows, cols = df.shape
    missing = df.isnull().sum().sum()

    tone = "clean and well-structured" if missing == 0 else "usable but requires preprocessing"

    story = f"""
    This dataset contains {rows} rows and {cols} features. 
    Overall data quality appears {tone}. The structure suggests suitability 
    for exploratory analysis and predictive modeling. Identified structural 
    patterns should be reviewed before production deployment.
    """

    return story.strip()