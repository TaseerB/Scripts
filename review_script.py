#!/usr/bin/env python3
"""
Employee Performance Review Generator
Asks questions about an employee's 6-month progress and generates a comprehensive review.
"""

import json
from datetime import datetime

class ReviewGenerator:
    def __init__(self):
        self.employee_data = {}
        self.responses = {}
        
    def ask_question(self, question, key, answer_type="text"):
        """Ask a question and store the response"""
        print(f"\n{question}")
        
        if answer_type == "rating":
            print("(Rate from 1-5: 1=Needs Improvement, 3=Meets Expectations, 5=Exceeds Expectations)")
            while True:
                try:
                    answer = int(input("Rating: "))
                    if 1 <= answer <= 5:
                        break
                    print("Please enter a number between 1 and 5")
                except ValueError:
                    print("Please enter a valid number")
        elif answer_type == "yes/no":
            while True:
                answer = input("Answer (yes/no): ").lower().strip()
                if answer in ['yes', 'no', 'y', 'n']:
                    answer = answer.startswith('y')
                    break
                print("Please answer yes or no")
        else:
            answer = input("Answer: ").strip()
        
        self.responses[key] = answer
        return answer
    
    def collect_employee_info(self):
        """Collect basic employee information"""
        print("=" * 60)
        print("EMPLOYEE PERFORMANCE REVIEW - 6 MONTH ASSESSMENT")
        print("=" * 60)
        
        self.employee_data['name'] = input("\nEmployee Name: ").strip()
        self.employee_data['position'] = input("Position/Role: ").strip()
        self.employee_data['department'] = input("Department: ").strip()
        self.employee_data['review_date'] = datetime.now().strftime("%B %d, %Y")
        
    def collect_performance_data(self):
        """Ask questions about performance over 6 months"""
        print("\n" + "=" * 60)
        print("PERFORMANCE ASSESSMENT")
        print("=" * 60)
        
        # Technical/Job Performance
        self.ask_question(
            "How would you rate their overall technical/job performance?",
            "technical_rating",
            "rating"
        )
        
        self.ask_question(
            "What were their key accomplishments in the past 6 months? (List major wins, projects completed, goals achieved)",
            "accomplishments",
            "text"
        )
        
        self.ask_question(
            "What technical skills or competencies have they demonstrated well?",
            "strengths",
            "text"
        )
        
        # Growth and Development
        self.ask_question(
            "Have they shown improvement or growth in any specific areas?",
            "growth_areas",
            "text"
        )
        
        self.ask_question(
            "Did they complete any training, certifications, or learning initiatives?",
            "training",
            "text"
        )
        
        # Collaboration and Teamwork
        self.ask_question(
            "How would you rate their teamwork and collaboration?",
            "teamwork_rating",
            "rating"
        )
        
        self.ask_question(
            "How do they work with colleagues? (Communication, helping others, team dynamics)",
            "collaboration_notes",
            "text"
        )
        
        # Initiative and Ownership
        self.ask_question(
            "Have they taken initiative beyond their core responsibilities?",
            "initiative",
            "yes/no"
        )
        
        if self.responses['initiative']:
            self.ask_question(
                "Describe examples of initiative or ownership they've shown:",
                "initiative_examples",
                "text"
            )
        
        # Challenges and Areas for Improvement
        self.ask_question(
            "What challenges did they face in the past 6 months?",
            "challenges",
            "text"
        )
        
        self.ask_question(
            "What areas need improvement or development?",
            "improvement_areas",
            "text"
        )
        
        # Goals and Future
        self.ask_question(
            "What goals should they focus on for the next 6 months?",
            "future_goals",
            "text"
        )
        
        self.ask_question(
            "Any additional comments or observations?",
            "additional_comments",
            "text"
        )
    
    def generate_review(self):
        """Generate a formatted performance review"""
        review = f"""
{'=' * 80}
EMPLOYEE PERFORMANCE REVIEW
6-Month Assessment Period
{'=' * 80}

EMPLOYEE INFORMATION
--------------------
Name:           {self.employee_data['name']}
Position:       {self.employee_data['position']}
Department:     {self.employee_data['department']}
Review Date:    {self.employee_data['review_date']}


PERFORMANCE SUMMARY
-------------------
Overall Technical Performance: {self.responses['technical_rating']}/5
Teamwork & Collaboration:      {self.responses['teamwork_rating']}/5


KEY ACCOMPLISHMENTS
-------------------
{self.responses['accomplishments']}


DEMONSTRATED STRENGTHS
----------------------
{self.responses['strengths']}


GROWTH AND DEVELOPMENT
----------------------
{self.responses['growth_areas']}

Training & Learning Initiatives:
{self.responses['training']}


COLLABORATION AND TEAMWORK
--------------------------
{self.responses['collaboration_notes']}
"""
        
        if self.responses['initiative']:
            review += f"""

INITIATIVE AND OWNERSHIP
------------------------
{self.responses.get('initiative_examples', 'Demonstrated initiative beyond core responsibilities.')}
"""
        
        review += f"""

CHALLENGES ENCOUNTERED
----------------------
{self.responses['challenges']}


AREAS FOR IMPROVEMENT
---------------------
{self.responses['improvement_areas']}


GOALS FOR NEXT 6 MONTHS
-----------------------
{self.responses['future_goals']}


ADDITIONAL COMMENTS
-------------------
{self.responses['additional_comments']}


{'=' * 80}
Overall Assessment: {self._get_overall_assessment()}
{'=' * 80}
"""
        
        return review
    
    def _get_overall_assessment(self):
        """Generate overall assessment based on ratings"""
        avg_rating = (self.responses['technical_rating'] + 
                     self.responses['teamwork_rating']) / 2
        
        if avg_rating >= 4.5:
            return "EXCEPTIONAL PERFORMANCE - Consistently exceeds expectations"
        elif avg_rating >= 3.5:
            return "STRONG PERFORMANCE - Exceeds expectations in key areas"
        elif avg_rating >= 2.5:
            return "MEETS EXPECTATIONS - Solid performance with room for growth"
        else:
            return "NEEDS IMPROVEMENT - Requires focused development"
    
    def save_review(self, filename=None):
        """Save review to file"""
        if filename is None:
            safe_name = self.employee_data['name'].replace(' ', '_').lower()
            filename = f"review_{safe_name}_{datetime.now().strftime('%Y%m%d')}.txt"
        
        review_text = self.generate_review()
        
        with open(filename, 'w') as f:
            f.write(review_text)
        
        return filename
    
    def run(self):
        """Main execution flow"""
        self.collect_employee_info()
        self.collect_performance_data()
        
        print("\n" + "=" * 60)
        print("Generating review...")
        print("=" * 60)
        
        review = self.generate_review()
        print(review)
        
        save = input("\nWould you like to save this review to a file? (yes/no): ").lower().strip()
        if save.startswith('y'):
            filename = self.save_review()
            print(f"\n✓ Review saved to: {filename}")
        
        # Also save raw data as JSON
        data_file = f"review_data_{self.employee_data['name'].replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d')}.json"
        with open(data_file, 'w') as f:
            json.dump({
                'employee': self.employee_data,
                'responses': self.responses
            }, f, indent=2)
        print(f"✓ Raw data saved to: {data_file}")


if __name__ == "__main__":
    print("\n🎯 Welcome to the Performance Review Generator!\n")
    print("This tool will guide you through creating a comprehensive")
    print("6-month performance review by asking targeted questions.\n")
    
    generator = ReviewGenerator()
    generator.run()
    
    print("\n✓ Review generation complete!")
    print("\nTip: You can edit the generated text file to add more detail or adjust wording.\n")