# -*- coding: utf-8 -*-
"""
LAKE TOBA ITINERARY GENERATOR - MINIMAL VERSION
Core itinerary generation logic without extras
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from geopy.distance import geodesic
from datetime import datetime
import ast
import warnings
warnings.filterwarnings('ignore')

class TobaItineraryGenerator:
    """Minimal itinerary generator for Lake Toba tourism"""
    
    def __init__(self):
        self.df_wisata = None
        self.df_akomodasi = None
        self.data_processed = None
        self.user_data = None
        # Define top 8 activities based on your data
        self.top_activities = [
            'pemandangan', 'fotografi', 'santai', 'camping', 
            'trekking', 'berenang', 'piknik', 'aktivitas air'
        ]
        
    def get_activity_categories(self):
        """Get activity categories with top 8 + others option"""
        activities = {
            '1': 'pemandangan',
            '2': 'fotografi', 
            '3': 'santai',
            '4': 'camping',
            '5': 'trekking',
            '6': 'berenang',
            '7': 'piknik',
            '8': 'aktivitas air',
            '9': 'lain-lain'
        }
        return activities
        
    def process_user_activities(self, selected_activities):
        """Process and normalize user activity selections"""
        processed_activities = []
        
        for activity in selected_activities:
            activity = activity.strip().lower()
            
            # Check if it's one of the top 8 activities
            if activity in self.top_activities:
                processed_activities.append(activity)
            else:
                # Group under 'lain-lain' if not in top 8
                if 'lain-lain' not in processed_activities:
                    processed_activities.append('lain-lain')
        
        return processed_activities if processed_activities else ['nature']
        
    def get_user_input(self):
        """Get user input with validation"""
        print("LAKE TOBA ITINERARY GENERATOR")
        print("=" * 40)
        
        # Date input
        while True:
            try:
                start_date_str = input("Start date (dd-mm-yy): ").strip()
                end_date_str = input("End date (dd-mm-yy): ").strip()
                
                start_date = datetime.strptime(start_date_str, "%d-%m-%y")
                end_date = datetime.strptime(end_date_str, "%d-%m-%y")
                
                if end_date < start_date:
                    print("Error: End date must be after start date!")
                    continue
                    
                num_days = (end_date - start_date).days + 1
                break
            except ValueError:
                print("Error: Invalid date format! Use dd-mm-yy")
        
        # Number of people
        while True:
            try:
                num_people = int(input("Number of people: "))
                if num_people <= 0:
                    print("Error: Number must be greater than 0!")
                    continue
                break
            except ValueError:
                print("Error: Please enter a valid number!")
        
        # Budget category
        print("\nBudget Category:")
        print("1. <1 million IDR")
        print("2. 1-2 million IDR") 
        print("3. >2 million IDR")
        
        while True:
            try:
                choice = int(input("Choose budget (1-3): "))
                if choice == 1:
                    budget_category = "<1 million"
                    budget_estimate = 1000000
                elif choice == 2:
                    budget_category = "1-2 million"
                    budget_estimate = 2000000
                elif choice == 3:
                    budget_category = ">2 million"
                    budget_estimate = 5000000
                else:
                    print("Error: Invalid choice!")
                    continue
                break
            except ValueError:
                print("Error: Please enter 1, 2, or 3!")
        
        # Activities - show top 8 + others option
        print("\nSelect Activities (you can choose multiple):")
        activity_categories = self.get_activity_categories()
        
        for key, value in activity_categories.items():
            print(f"{key}. {value.title()}")
        
        print("\nEnter activity numbers separated by commas (e.g., 1,3,5)")
        print("Or enter custom activities separated by commas")
        
        activities_input = input("Your choice: ").strip()
        
        # Process input - could be numbers or custom text
        selected_activities = []
        
        # Check if input contains numbers (referring to the menu)
        if any(char.isdigit() for char in activities_input):
            # Process as menu selections
            try:
                choices = [choice.strip() for choice in activities_input.split(",")]
                for choice in choices:
                    if choice in activity_categories:
                        selected_activities.append(activity_categories[choice])
                    elif choice.isdigit() and 1 <= int(choice) <= 9:
                        selected_activities.append(activity_categories[str(choice)])
            except:
                pass
        
        # If no valid menu selections or custom input, process as text
        if not selected_activities:
            custom_activities = [act.strip().lower() for act in activities_input.split(",") if act.strip()]
            selected_activities = custom_activities
        
        # Process and normalize activities
        activities = self.process_user_activities(selected_activities)
        
        print(f"Selected activities: {', '.join(activities)}")
        
        self.user_data = {
            'start_date': start_date,
            'end_date': end_date,
            'num_days': num_days,
            'num_people': num_people,
            'budget_category': budget_category,
            'budget_estimate': budget_estimate,
            'activities': activities
        }
        
        return self.user_data

    def load_data(self):
        """Load datasets from GitHub"""
        print("\nLoading data...")
        
        try:
            urls = {
                'processed': 'https://raw.githubusercontent.com/kevinyogap/data/refs/heads/main/data_ProcessedZ.csv',
                'wisata': 'https://raw.githubusercontent.com/kevinyogap/data/refs/heads/main/data_wisata_toba_score.csv',
                'akomodasi': 'https://raw.githubusercontent.com/kevinyogap/data/refs/heads/main/data_akomodasi_danau_toba_v2.csv'
            }
            
            df_P = pd.read_csv(urls['processed'])
            self.df_wisata = pd.read_csv(urls['wisata'])
            self.df_akomodasi = pd.read_csv(urls['akomodasi'])
            
            self.data_processed = pd.merge(self.df_wisata, df_P, on='title', how='inner')
            
            print(f"Data loaded: {len(self.df_wisata)} tourism spots, {len(self.df_akomodasi)} hotels")
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def get_tourism_recommendations(self, top_n=10):
        """Get tourism recommendations using TF-IDF"""
        print(f"\nFinding {top_n} recommendations...")

        user_query = ' '.join(self.user_data['activities'])
        
        try:
            data_processed = self.data_processed.copy()
            
            # Prepare category data
            data_processed['kategori'] = data_processed['kategori_token'].apply(
                lambda x: ast.literal_eval(x)[0] if pd.notna(x) and x != '[]' else 'general'
            )

            # TF-IDF vectorization
            vectorizer = TfidfVectorizer(lowercase=True, stop_words=None)
            
            all_texts = list(data_processed['tags_joined'].fillna('')) + [user_query]
            vectorizer.fit(all_texts)

            tfidf_wisata = vectorizer.transform(data_processed['tags_joined'].fillna(''))
            tfidf_user = vectorizer.transform([user_query])

            # Calculate similarity
            similarities = cosine_similarity(tfidf_user, tfidf_wisata).flatten()
            max_similarity = np.max(similarities)

            if max_similarity > 0:
                data_processed['similarity_score'] = similarities
                recommendations_df = data_processed.nlargest(top_n, 'similarity_score')
            else:
                # Fallback to score-based recommendations
                if 'score' in data_processed.columns:
                    recommendations_df = data_processed.nlargest(top_n, 'score')
                else:
                    recommendations_df = data_processed.sample(min(top_n, len(data_processed)))

            # Format output
            recommendations = []
            for idx, row in recommendations_df.iterrows():
                score = row.get('similarity_score', row.get('score', 0))
                recommendations.append((row['title'], row['kategori'], score))

            print(f"Top 5 recommendations:")
            for i, (title, category, score) in enumerate(recommendations[:5], 1):
                print(f"  {i}. {title} ({category})")

            return recommendations

        except Exception as e:
            print(f"Error in recommendations: {e}")
            return []

    def find_nearest_hotel(self, recommendations):
        """Find nearest hotel from top tourism spot"""
        print(f"\nFinding nearest hotel...")
        
        if not recommendations:
            return None, None
        
        top_spot_name = recommendations[0][0]
        top_spot = self.df_wisata[self.df_wisata['title'] == top_spot_name]
        
        if top_spot.empty:
            return None, None
        
        top_spot = top_spot.iloc[0]
        
        # Calculate distance to all hotels
        df_akomodasi_copy = self.df_akomodasi.copy()
        df_akomodasi_copy['distance'] = df_akomodasi_copy.apply(
            lambda row: geodesic(
                (row['latitude'], row['longitude']),
                (top_spot['latitude'], top_spot['longitude'])
            ).km,
            axis=1
        )
        
        nearest_hotel = df_akomodasi_copy.sort_values('distance').iloc[0]
        
        print(f"Selected hotel: {nearest_hotel['name']} ({nearest_hotel['distance']:.1f} km)")
        
        return nearest_hotel, top_spot

    def select_tourism_spots(self, recommendations, starting_hotel, max_spots=5):
        """Select nearest tourism spots"""
        print(f"\nSelecting {max_spots} tourism spots...")
        
        recommendation_names = [item[0] for item in recommendations]
        top_df = self.df_wisata[self.df_wisata['title'].isin(recommendation_names)]
        
        selected_spots = []
        current_location = (starting_hotel['latitude'], starting_hotel['longitude'])
        
        for i in range(min(max_spots, len(top_df))):
            candidates = top_df[~top_df['title'].isin(selected_spots)].copy()
            
            if candidates.empty:
                break
            
            # Calculate distance from current location
            candidates['distance'] = candidates.apply(
                lambda row: geodesic(current_location, (row['latitude'], row['longitude'])).km,
                axis=1
            )
            
            selected = candidates.sort_values('distance').iloc[0]
            selected_spots.append(selected['title'])
            current_location = (selected['latitude'], selected['longitude'])
        
        print(f"Selected spots: {', '.join(selected_spots)}")
        return selected_spots

    def validate_time_constraint(self, hotel, tourism_spots, max_hours=8):
        """Validate that total trip time is within constraint"""
        print(f"\nValidating time constraint ({max_hours} hours max)...")
        
        valid_spots = tourism_spots.copy()
        
        while valid_spots:
            # Calculate total distance
            total_distance = 0
            current_location = (hotel['latitude'], hotel['longitude'])
            
            for spot in valid_spots:
                row = self.df_wisata[self.df_wisata['title'] == spot]
                if not row.empty:
                    row = row.iloc[0]
                    distance = geodesic(current_location, (row['latitude'], row['longitude'])).km
                    total_distance += distance
                    current_location = (row['latitude'], row['longitude'])
            
            # Add return distance to hotel
            total_distance += geodesic(current_location, (hotel['latitude'], hotel['longitude'])).km
            
            # Calculate total time (travel + visit time)
            travel_time = total_distance / 30  # 30 km/h average speed
            visit_time = len(valid_spots) * 1.5  # 1.5 hours per spot
            total_time = travel_time + visit_time
            
            if total_time <= max_hours:
                print(f"Valid itinerary: {len(valid_spots)} spots, {total_time:.1f} hours")
                break
            else:
                print(f"Too long ({total_time:.1f}h), removing: {valid_spots[-1]}")
                valid_spots.pop()
        
        return valid_spots

    def calculate_costs(self, hotel, tourism_spots):
        """Calculate itinerary costs"""
        # Tourism costs
        tourism_df = self.df_wisata[self.df_wisata['title'].isin(tourism_spots)]
        entrance_cost = tourism_df['biaya_masuk'].fillna(0).sum() if 'biaya_masuk' in tourism_df.columns else 0
        parking_cost = tourism_df['biaya_parkir_mobil'].fillna(0).sum() if 'biaya_parkir_mobil' in tourism_df.columns else 0
        tourism_cost = entrance_cost + parking_cost
        
        # Hotel cost
        hotel_cost = hotel.get('price', 0) * max(self.user_data['num_days'] - 1, 1)
        
        total_cost = tourism_cost + hotel_cost
        budget = self.user_data['budget_estimate']
        within_budget = total_cost <= budget
        
        return {
            'tourism_cost': tourism_cost,
            'hotel_cost': hotel_cost,
            'total_cost': total_cost,
            'budget': budget,
            'within_budget': within_budget
        }

    def create_itinerary(self, hotel, tourism_spots):
        """Create final itinerary"""
        print(f"\nCreating itinerary...")
        
        # Calculate costs
        costs = self.calculate_costs(hotel, tourism_spots)
        
        # Create schedule
        print(f"\n" + "=" * 50)
        print(f"LAKE TOBA ITINERARY")
        print(f"=" * 50)
        print(f"Period: {self.user_data['start_date'].strftime('%d-%m-%Y')} - {self.user_data['end_date'].strftime('%d-%m-%Y')}")
        print(f"Days: {self.user_data['num_days']}")
        print(f"People: {self.user_data['num_people']}")
        print(f"Budget: {self.user_data['budget_category']}")
        print(f"Activities: {', '.join(self.user_data['activities'])}")
        print(f"\nHOTEL: {hotel['name']}")
        print(f"\nTOURISM SPOTS:")
        for i, spot in enumerate(tourism_spots, 1):
            print(f"  {i}. {spot}")
        
        print(f"\nCOST BREAKDOWN:")
        print(f"  Tourism: IDR {costs['tourism_cost']:,}")
        print(f"  Hotel: IDR {costs['hotel_cost']:,}")
        print(f"  Total: IDR {costs['total_cost']:,}")
        print(f"  Budget: IDR {costs['budget']:,}")
        
        if costs['within_budget']:
            print(f"  Status: Within budget ✓")
        else:
            print(f"  Status: Over budget ✗")
        
        return {
            'hotel': hotel,
            'tourism_spots': tourism_spots,
            'costs': costs
        }

    def find_cheaper_hotel_near_route(self, tourism_spots, max_distance_km=10):
        """
        Find the cheapest hotel near any of the tourism spots within max_distance_km.
        Returns the hotel row or None if not found.
        """
        if not tourism_spots:
            return None

        # Get coordinates of all spots in the route
        spots_df = self.df_wisata[self.df_wisata['title'].isin(tourism_spots)]
        spot_coords = spots_df[['latitude', 'longitude']].values

        # For each hotel, check if it's close to any spot
        hotels = self.df_akomodasi.copy()
        hotels['min_distance'] = hotels.apply(
            lambda row: min(
                geodesic((row['latitude'], row['longitude']), (lat, lon)).km
                for lat, lon in spot_coords
            ),
            axis=1
        )
        # Filter hotels within max_distance_km to any spot
        nearby_hotels = hotels[hotels['min_distance'] <= max_distance_km]
        if nearby_hotels.empty:
            # fallback: use all hotels
            nearby_hotels = hotels

        # Return the cheapest hotel (by price)
        cheapest = nearby_hotels.sort_values('price').iloc[0]
        return cheapest

    def run_generator(self):
        """Main program execution"""
        print("Starting Lake Toba Itinerary Generator...\n")
        try:
            # 1. Get user input
            self.get_user_input()
            # 2. Load data
            if not self.load_data():
                return None
            # 3. Get recommendations
            recommendations = self.get_tourism_recommendations()
            if not recommendations:
                print("Error: Cannot create recommendations!")
                return None
            # 4. Find hotel
            hotel, _ = self.find_nearest_hotel(recommendations)
            if hotel is None:
                print("Error: Cannot find suitable hotel!")
                return None
            # 5. Select tourism spots
            tourism_spots = self.select_tourism_spots(recommendations, hotel)
            if not tourism_spots:
                print("Error: No tourism spots selected!")
                return None
            # 6. Validate time constraints
            valid_spots = self.validate_time_constraint(hotel, tourism_spots)
            if not valid_spots:
                print("Error: No spots meet time criteria!")
                return None

            # 7. Create itinerary and check budget
            result = self.create_itinerary(hotel, valid_spots)
            if not result['costs']['within_budget']:
                # Try to find a cheaper hotel near the route
                cheaper_hotel = self.find_cheaper_hotel_near_route(valid_spots)
                if cheaper_hotel is not None and cheaper_hotel['price'] < hotel['price']:
                    # Recalculate with cheaper hotel
                    new_result = self.create_itinerary(cheaper_hotel, valid_spots)
                    if new_result['costs']['within_budget']:
                        print("\nCheaper hotel found and used to fit the budget.")
                        result = new_result
                    else:
                        print("\nCheaper hotel found, but still over budget.")
                else:
                    print("\nNo cheaper hotel found nearby or all are more expensive.")

            print("\nITINERARY CREATED.")
            return result

        except KeyboardInterrupt:
            print(f"\nProgram interrupted by user.")
            return None
        except Exception as e:
            print(f"Error: {e}")
            return None

# Run the generator
if __name__ == "__main__":
    generator = TobaItineraryGenerator()
    result = generator.run_generator()