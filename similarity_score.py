def calculate_composite_score(scores):
    # Unpacking scores
    lcs, jaro_winkler, similarity, difflib = [float(score) for score in scores]

    # Base score is the similarity score
    base_score = similarity

    if base_score == 1.0:
        return "1.00000"

    # Finding the average of the high-performing metrics for adjustment consideration
    high_performance_scores = [lcs, jaro_winkler, difflib]
    high_performance_average = sum(high_performance_scores) / len(high_performance_scores)

    # Define the adjustment based on the base score
    if base_score == 0:
        scale_factor = 0.05  # Minimal influence when no base confidence
    elif base_score < 0.1:
        scale_factor = 0.1
    elif base_score < 0.2:
        scale_factor = 0.15
    elif base_score < 0.3:
        scale_factor = 0.2
    elif base_score < 0.4:
        scale_factor = 0.25
    elif base_score < 0.5:
        scale_factor = 0.3
    else:
        scale_factor = 0.35  # Maximum influence when base score is reasonably high

    # Calculate the adjustment
    adjustment = (high_performance_average - base_score*0.5) * scale_factor
    adjustment = max(adjustment, 0)  # Ensure that adjustment is not negative

    adjusted_score = base_score + adjustment
    # Minimum threshold for high fuzzy metrics
    if high_performance_average > 0.75:
        adjusted_score = max(adjusted_score, 0.25)

    # Apply the adjustment and ensure it does not exceed 1
    composite_score = min(adjusted_score, 1.0)
    
    return "{:.5f}".format(composite_score)