from dataclasses import dataclass, field

from .tutor import clean_tutor_row, split_hours

#capacity should be a snapshot of the current capacity info. We want to update capacity (by recalculating it from the pool of tutors), get it (to see how many hours we have), what else?


@dataclass
class Capacity:
    capacity: dict = field(default_factory=dict)  # {subject: {gender: weekly hours}}

    def update_capacity(self, tutors, demand):
        """Rebuild capacity from (hours, subjects, gender) rows.

        Single-subject tutors count in full. Multi-subject tutors' hours are split in
        proportion to the demand still unmet after single-subject tutors are counted.
        demand: {subject: {gender: weekly hours}}, e.g. the forecast for next month.
        """
        rows = [clean_tutor_row(*row) for row in tutors]
        demand = {s.lower(): d for s, d in demand.items()}
        self.capacity = {}

        for hours, subjects, gender in rows:
            if len(subjects) == 1:
                self.add_hours(subjects[0], gender, hours)

        # work out every split before adding any, so each uses the single-subject totals
        splits = [(split_hours(hours, self._unmet_demand(subjects, gender, demand)), gender)
                  for hours, subjects, gender in rows if len(subjects) > 1]
        for split, gender in splits:
            for subject, hours in split.items():
                self.add_hours(subject, gender, hours)

    def _unmet_demand(self, subjects, gender, demand):
        return {s: max(demand.get(s, {}).get(gender, 0) - self.get_subject_capacity(s, gender), 0)
                for s in subjects}

    def add_hours(self, subject, gender, hours):
        """Add hours to a subject, e.g. for a new recruit."""
        if gender not in (0, 1):
            raise ValueError(f"gender must be 0 or 1, got {gender!r}")
        by_gender = self.capacity.setdefault(subject.lower(), {0: 0, 1: 0})
        by_gender[gender] += hours

    def use_hours(self, subject, gender, hours):
        """Take hours from a subject, e.g. when a new client starts."""
        available = self.get_subject_capacity(subject, gender)
        if hours > available:
            raise ValueError(f"cannot use {hours} {subject} hours, only {available} available")
        self.add_hours(subject, gender, -hours)

    def get_subject_capacity(self, subject, gender):
        return self.capacity.get(subject.lower(), {}).get(gender, 0)
