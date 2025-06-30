# -*- coding: utf-8 -*-
"""
LAKE TOBA ITINERARY GENERATOR - FLASK API VERSION
RESTful API for mobile application integration
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from geopy.distance import geodesic
from datetime import datetime
import ast
import warnings
import traceback
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)  # Enable CORS for mobile app access

class TobaItineraryService:
    """Service class for Lake Toba itinerary generation"""
    
    def __init__(self):
        self.df_wisata = None
        self.df_akomodasi = None
        self.data_processed = None
        self.data_loaded = False
        
        # Define top 8 activities
        self.top_activities = [
            'pemandangan', 'fotografi', 'santai', 'camping', 
            'trekking', 'berenang', 'piknik', 'aktivitas air'
        ]
        
        # Load data on initialization
        self.load_data()
        
    def get_activity_categories(self):
        """Get available activity categories"""
        return {
            'pemandangan': 'Scenic Views',
            'fotografi': 'Photography', 
            'santai': 'Relaxation',
            'camping': 'Camping',
            'trekking': 'Trekking',
            'berenang': 'Swimming',
            'piknik': 'Picnic',
            'aktivitas air': 'Water Activities',
            'lain-lain': 'Others'
        }
        
    def load_data(self):
        """Load datasets from GitHub"""
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
            self.data_loaded = True
            
            print(f"Data loaded: {len(self.df_wisata)} tourism spots, {len(self.df_akomodasi)} hotels")
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            self.data_loaded = False
            return False
    
    def process_user_activities(self, selected_activities):
        """Process and normalize user activity selections"""
        processed_activities = []
        
        for activity in selected_activities:
            activity = activity.strip().lower()
            
            if activity in self.top_activities:
                processed_activities.append(activity)
            else:
                if 'lain-lain' not in processed_activities:
                    processed_activities.append('lain-lain')
        
        return processed_activities if processed_activities else ['pemandangan']
    
    def get_tourism_recommendations(self, activities, top_n=10):
        """Get tourism recommendations using TF-IDF"""
        if not self.data_loaded:
            raise Exception("Data not loaded")
            
        user_query = ' '.join(activities)
        
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

            # Format recommendations
            recommendations = []
            for idx, row in recommendations_df.iterrows():
                score = row.get('similarity_score', row.get('score', 0))
                recommendations.append({
                    'title': row['title'],
                    'category': row['kategori'],
                    'score': float(score),
                    'latitude': float(row['latitude']),
                    'longitude': float(row['longitude']),
                    'entrance_fee': float(row.get('biaya_masuk', 0)),
                    'parking_fee': float(row.get('biaya_parkir_mobil', 0))
                })

            return recommendations

        except Exception as e:
            raise Exception(f"Error in recommendations: {e}")
    
    def find_hotels_near_spots(self, tourism_spots, max_distance_km=15):
        """Find hotels near tourism spots"""
        if not tourism_spots:
            return []
            
        # Get coordinates of tourism spots
        spot_names = [spot['title'] for spot in tourism_spots]
        spots_df = self.df_wisata[self.df_wisata['title'].isin(spot_names)]
        
        if spots_df.empty:
            return []
        
        spot_coords = spots_df[['latitude', 'longitude']].values
        
        # Find hotels within distance
        hotels = self.df_akomodasi.copy()
        hotels['min_distance'] = hotels.apply(
            lambda row: min(
                geodesic((row['latitude'], row['longitude']), (lat, lon)).km
                for lat, lon in spot_coords
            ),
            axis=1
        )
        
        nearby_hotels = hotels[hotels['min_distance'] <= max_distance_km]
        
        # Format hotel data
        hotel_list = []
        for idx, row in nearby_hotels.iterrows():
            hotel_list.append({
                'name': row['name'],
                'price': float(row.get('price', 0)),
                'latitude': float(row['latitude']),
                'longitude': float(row['longitude']),
                'distance_to_spots': float(row['min_distance']),
                # 'rating': float(row.get('rating', 0)) if 'rating' in row else 0
                'rating': float(str(row.get('rating', 0)).replace(',', '.')) if 'rating' in row else 0

            })
        
        # Sort by distance then by price
        hotel_list.sort(key=lambda x: (x['distance_to_spots'], x['price']))
        
        return hotel_list
    
    def calculate_trip_cost(self, tourism_spots, hotel, num_days, num_people):
        """Calculate total trip cost"""
        # Tourism costs
        entrance_cost = sum(spot.get('entrance_fee', 0) for spot in tourism_spots) * num_people
        parking_cost = sum(spot.get('parking_fee', 0) for spot in tourism_spots)
        tourism_cost = entrance_cost + parking_cost
        
        # Hotel cost (num_days - 1 nights)
        hotel_nights = max(num_days - 1, 1)
        hotel_cost = hotel.get('price', 0) * hotel_nights
        
        # Estimated transport cost (rough estimate)
        transport_cost = 200000 * num_days  # IDR 200k per day
        
        total_cost = tourism_cost + hotel_cost + transport_cost
        
        return {
            'tourism_cost': tourism_cost,
            'hotel_cost': hotel_cost,
            'transport_cost': transport_cost,
            'total_cost': total_cost,
            'per_person_cost': total_cost / num_people if num_people > 0 else total_cost
        }
    
    def optimize_route(self, tourism_spots, hotel, max_hours=8):
        """Optimize tourism route based on distance and time constraints"""
        if not tourism_spots:
            return []
        
        # Get spot coordinates
        spot_data = []
        for spot in tourism_spots:
            spot_data.append({
                'title': spot['title'],
                'latitude': spot['latitude'],
                'longitude': spot['longitude'],
                'entrance_fee': spot.get('entrance_fee', 0),
                'parking_fee': spot.get('parking_fee', 0)
            })
        
        # Simple greedy optimization: start from hotel, visit nearest unvisited spot
        optimized_route = []
        current_location = (hotel['latitude'], hotel['longitude'])
        remaining_spots = spot_data.copy()
        
        while remaining_spots:
            # Find nearest spot
            nearest_spot = None
            min_distance = float('inf')
            
            for spot in remaining_spots:
                distance = geodesic(current_location, (spot['latitude'], spot['longitude'])).km
                if distance < min_distance:
                    min_distance = distance
                    nearest_spot = spot
            
            if nearest_spot:
                optimized_route.append(nearest_spot)
                remaining_spots.remove(nearest_spot)
                current_location = (nearest_spot['latitude'], nearest_spot['longitude'])
            else:
                break
        
        # Validate time constraint
        total_distance = 0
        current_location = (hotel['latitude'], hotel['longitude'])
        
        for spot in optimized_route:
            distance = geodesic(current_location, (spot['latitude'], spot['longitude'])).km
            total_distance += distance
            current_location = (spot['latitude'], spot['longitude'])
        
        # Add return distance to hotel
        total_distance += geodesic(current_location, (hotel['latitude'], hotel['longitude'])).km
        
        # Calculate total time
        travel_time = total_distance / 30  # 30 km/h average speed
        visit_time = len(optimized_route) * 1.5  # 1.5 hours per spot
        total_time = travel_time + visit_time
        
        # Remove spots if over time limit
        while total_time > max_hours and optimized_route:
            optimized_route.pop()
            # Recalculate time
            total_distance = 0
            current_location = (hotel['latitude'], hotel['longitude'])
            
            for spot in optimized_route:
                distance = geodesic(current_location, (spot['latitude'], spot['longitude'])).km
                total_distance += distance
                current_location = (spot['latitude'], spot['longitude'])
            
            if optimized_route:
                total_distance += geodesic(current_location, (hotel['latitude'], hotel['longitude'])).km
            
            travel_time = total_distance / 30
            visit_time = len(optimized_route) * 1.5
            total_time = travel_time + visit_time
        
        return {
            'route': optimized_route,
            'total_distance_km': total_distance,
            'total_time_hours': total_time,
            'travel_time_hours': travel_time,
            'visit_time_hours': visit_time
        }

# Initialize service
toba_service = TobaItineraryService()

# API Routes
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'data_loaded': toba_service.data_loaded,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/activities', methods=['GET'])
def get_activities():
    """Get available activity categories"""
    try:
        activities = toba_service.get_activity_categories()
        return jsonify({
            'success': True,
            'data': activities
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    """Get tourism recommendations based on activities"""
    try:
        data = request.get_json()
        
        # Validate input
        if not data or 'activities' not in data:
            return jsonify({
                'success': False,
                'error': 'Activities are required'
            }), 400
        
        activities = data['activities']
        top_n = data.get('top_n', 10)
        
        # Process activities
        processed_activities = toba_service.process_user_activities(activities)
        
        # Get recommendations
        recommendations = toba_service.get_tourism_recommendations(processed_activities, top_n)
        
        return jsonify({
            'success': True,
            'data': {
                'processed_activities': processed_activities,
                'recommendations': recommendations,
                'total_count': len(recommendations)
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/hotels', methods=['POST'])
def get_hotels():
    """Get hotels near tourism spots"""
    try:
        data = request.get_json()
        
        if not data or 'tourism_spots' not in data:
            return jsonify({
                'success': False,
                'error': 'Tourism spots are required'
            }), 400
        
        tourism_spots = data['tourism_spots']
        max_distance = data.get('max_distance_km', 15)
        
        hotels = toba_service.find_hotels_near_spots(tourism_spots, max_distance)
        
        return jsonify({
            'success': True,
            'data': {
                'hotels': hotels,
                'total_count': len(hotels)
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/calculate-cost', methods=['POST'])
def calculate_cost():
    """Calculate trip cost"""
    try:
        data = request.get_json()
        
        required_fields = ['tourism_spots', 'hotel', 'num_days', 'num_people']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'{field} is required'
                }), 400
        
        costs = toba_service.calculate_trip_cost(
            data['tourism_spots'],
            data['hotel'],
            data['num_days'],
            data['num_people']
        )
        
        return jsonify({
            'success': True,
            'data': costs
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/optimize-route', methods=['POST'])
def optimize_route():
    """Optimize tourism route"""
    try:
        data = request.get_json()
        
        required_fields = ['tourism_spots', 'hotel']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'{field} is required'
                }), 400
        
        max_hours = data.get('max_hours', 8)
        
        route_info = toba_service.optimize_route(
            data['tourism_spots'],
            data['hotel'],
            max_hours
        )
        
        return jsonify({
            'success': True,
            'data': route_info
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/check-budget', methods=['POST'])
def check_budget():
    """Check if trip is within budget"""
    try:
        data = request.get_json()
        
        required_fields = ['tourism_spots', 'hotel', 'num_days', 'num_people', 'budget']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'{field} is required'
                }), 400
        
        costs = toba_service.calculate_trip_cost(
            data['tourism_spots'],
            data['hotel'],
            data['num_days'],
            data['num_people']
        )
        
        budget = data['budget']
        within_budget = costs['total_cost'] <= budget
        budget_difference = budget - costs['total_cost']
        
        return jsonify({
            'success': True,
            'data': {
                'costs': costs,
                'budget': budget,
                'within_budget': within_budget,
                'budget_difference': budget_difference,
                'budget_utilization_percent': (costs['total_cost'] / budget) * 100 if budget > 0 else 0
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/create-itinerary', methods=['POST'])
def create_full_itinerary():
    """Create complete itinerary"""
    try:
        data = request.get_json()
        
        required_fields = ['activities', 'num_days', 'num_people', 'budget']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'{field} is required'
                }), 400
        
        # Get recommendations
        processed_activities = toba_service.process_user_activities(data['activities'])
        recommendations = toba_service.get_tourism_recommendations(processed_activities, 10)
        
        if not recommendations:
            return jsonify({
                'success': False,
                'error': 'No tourism recommendations found'
            }), 404
        
        # Get top 5 spots
        top_spots = recommendations[:5]
        
        # Find hotels
        hotels = toba_service.find_hotels_near_spots(top_spots, 15)
        
        if not hotels:
            return jsonify({
                'success': False,
                'error': 'No hotels found near selected spots'
            }), 404
        
        # Select best hotel (cheapest among closest)
        selected_hotel = hotels[0]
        
        # Optimize route
        route_info = toba_service.optimize_route(top_spots, selected_hotel, 8)
        optimized_spots = route_info['route']
        
        # Calculate costs
        costs = toba_service.calculate_trip_cost(
            optimized_spots,
            selected_hotel,
            data['num_days'],
            data['num_people']
        )
        
        # Check budget
        budget = data['budget']
        within_budget = costs['total_cost'] <= budget
        
        # If over budget, try cheaper hotel
        if not within_budget and len(hotels) > 1:
            for hotel in hotels[1:]:
                if hotel['price'] < selected_hotel['price']:
                    test_costs = toba_service.calculate_trip_cost(
                        optimized_spots,
                        hotel,
                        data['num_days'],
                        data['num_people']
                    )
                    if test_costs['total_cost'] <= budget:
                        selected_hotel = hotel
                        costs = test_costs
                        within_budget = True
                        break
        
        return jsonify({
            'success': True,
            'data': {
                'itinerary': {
                    'activities': processed_activities,
                    'tourism_spots': optimized_spots,
                    'hotel': selected_hotel,
                    'route_info': route_info,
                    'num_days': data['num_days'],
                    'num_people': data['num_people']
                },
                'costs': costs,
                'budget': budget,
                'within_budget': within_budget,
                'budget_difference': budget - costs['total_cost']
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    print("Starting Lake Toba Itinerary API...")
    print("API Endpoints:")
    print("- GET  /api/health - Health check")
    print("- GET  /api/activities - Get activity categories")
    print("- POST /api/recommendations - Get tourism recommendations")
    print("- POST /api/hotels - Get hotels near spots")
    print("- POST /api/calculate-cost - Calculate trip cost")
    print("- POST /api/optimize-route - Optimize route")
    print("- POST /api/check-budget - Check budget")
    print("- POST /api/create-itinerary - Create complete itinerary")
    print("\nStarting server on http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)