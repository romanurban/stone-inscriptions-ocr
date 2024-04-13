class CompositeScoreCalculator:
    def __init__(self, scores):
        self.lcs, self.jaro_winkler, self.similarity, self.difflib = [float(score) for score in scores]
    
    def calculate(self):
        base_score = self.similarity

        if base_score == 1.0:
            return "1.00000"
        
        high_performance_scores = [self.lcs, self.jaro_winkler, self.difflib]
        high_performance_average = sum(high_performance_scores) / len(high_performance_scores)

        scale_factor = self._determine_scale_factor(base_score)
        adjustment = (high_performance_average - base_score * 0.5) * scale_factor
        adjustment = max(adjustment, 0)  # Ensure that adjustment is not negative

        adjusted_score = base_score + adjustment
        # Minimum threshold for high fuzzy metrics
        if high_performance_average > 0.75:
            adjusted_score = max(adjusted_score, 0.25)

        composite_score = min(adjusted_score, 1.0)
        return "{:.5f}".format(composite_score)

    def _determine_scale_factor(self, base_score):
        if base_score == 0:
            return 0.05
        elif base_score < 0.1:
            return 0.1
        elif base_score < 0.2:
            return 0.15
        elif base_score < 0.3:
            return 0.2
        elif base_score < 0.4:
            return 0.25
        elif base_score < 0.5:
            return 0.3
        else:
            return 0.35  # Maximum influence when base score is reasonably high
