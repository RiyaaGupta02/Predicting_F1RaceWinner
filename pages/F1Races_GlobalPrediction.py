# f1_race.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from Predicting_f1_pipeline import F1RacePredictionAPI, TRACK_FACTORS
import warnings
warnings.filterwarnings('ignore')

# Import SHAP with error handling
try:
    import shap
    print("✅ SHAP library loaded successfully")
except ImportError:
    print("❌ SHAP library not found. Install with: pip install shap")
    shap = None

st.set_page_config(page_title="🏎️ F1 Race Predictor", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF1E1E;
        text-align: center;
        margin-bottom: 2rem;
    }
    .winner-box {
        background-color: #FFD700;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .analysis-section {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Page title and description
st.markdown('<h1 class="main-header">🏁 F1 Race Winner Predictor</h1>', unsafe_allow_html=True)
st.markdown("### Enter race information below to predict the winner and analyze key factors")

# Initialize API with caching
@st.cache_resource
def load_api():
    try:
        return F1RacePredictionAPI()
    except Exception as e:
        st.error(f"Failed to load F1 Prediction API: {str(e)}")
        st.stop()

# Helper function to convert time format
def convert_time_to_seconds(time_str):
    """Convert time string (M:SS.sss) to seconds"""
    try:
        if not time_str or time_str.strip() == "":
            return None
        
        time_str = time_str.strip()
        
        # Handle different time formats
        if ":" in time_str:
            parts = time_str.split(":")
            if len(parts) == 2:
                minutes = int(parts[0])
                seconds = float(parts[1])
                return minutes * 60 + seconds
        else:
            # If no colon, assume it's just seconds
            return float(time_str)
    except (ValueError, IndexError) as e:
        st.error(f"Invalid time format: {time_str}. Expected format: M:SS.sss (e.g., 1:22.456)")
        return None

# FIXED safe_display_figure function with comprehensive error handling
def safe_display_figure(fig, error_message="Error displaying plot"):
    """Safely display matplotlib figure in Streamlit with proper memory management"""
    try:
        if fig is None:
            st.warning("Figure is None - plot was not generated")
            return False
            
        # Verify figure is valid matplotlib figure
        if not hasattr(fig, 'savefig'):
            st.warning("Invalid figure object - not a matplotlib figure")
            return False
            
        # Check if figure has any content
        if not fig.axes:
            st.warning("Figure has no content to display")
            plt.close(fig)
            return False
            
        # Display the figure
        st.pyplot(fig, clear_figure=True)
        
        # Important: close to free memory
        plt.close(fig)
        return True
        
    except Exception as e:
        st.error(f"{error_message}: {str(e)}")
        # Try to close the figure even if there's an error
        try:
            if fig is not None:
                plt.close(fig)
        except:
            pass
        return False

# Enhanced SHAP validation function
def validate_shap_data(shap_analysis):
    """Comprehensive validation of SHAP analysis data structure"""
    if not shap_analysis or not isinstance(shap_analysis, dict):
        return False, "SHAP analysis data is missing or invalid format"
    
    # Check for required keys
    expected_keys = ['winner_analysis', 'feature_importance_fig', 'heatmap_fig', 'summary_fig']
    available_keys = [key for key in expected_keys if key in shap_analysis]
    
    if not available_keys:
        return False, "No expected SHAP analysis components found"
    
    # Validate each component
    valid_components = []
    
    # Check winner analysis
    if 'winner_analysis' in shap_analysis:
        winner_data = shap_analysis['winner_analysis']
        if isinstance(winner_data, dict) and winner_data:
            valid_components.append('winner_analysis')
    
    # Check figures
    for fig_key in ['feature_importance_fig', 'heatmap_fig', 'summary_fig']:
        if fig_key in shap_analysis:
            fig = shap_analysis[fig_key]
            if fig is not None and hasattr(fig, 'savefig'):
                valid_components.append(fig_key)
    
    if not valid_components:
        return False, "All SHAP components are empty or invalid"
    
    return True, f"SHAP data is valid with components: {', '.join(valid_components)}"

# Function to create fallback analysis
def create_fallback_analysis(results_df, api):
    """Create basic analysis when SHAP fails"""
    try:
        fallback_analysis = {}
        
        # Basic feature importance from model if available
        if hasattr(api, 'model') and hasattr(api.model, 'feature_importances_'):
            feature_cols = getattr(api, 'feature_cols', [])
            if feature_cols:
                basic_importance = pd.DataFrame({
                    'Feature': feature_cols,
                    'Importance': api.model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fallback_analysis['basic_importance'] = basic_importance
        
        # Simple statistics
        if not results_df.empty:
            fallback_analysis['top_driver'] = results_df.iloc[0]['Driver']
            fallback_analysis['win_probability'] = results_df.iloc[0]['WinProbability']
            fallback_analysis['probability_range'] = {
                'min': results_df['WinProbability'].min(),
                'max': results_df['WinProbability'].max(),
                'mean': results_df['WinProbability'].mean()
            }
        
        return fallback_analysis
        
    except Exception as e:
        st.error(f"Error creating fallback analysis: {str(e)}")
        return {}

# Load API
api = load_api()

# Sidebar for race configuration
with st.sidebar:
    st.header("🏎️ Race Configuration")
    
    # Track selection
    track_name = st.selectbox("🏁 Select Grand Prix", list(TRACK_FACTORS.keys()))
    track_factors = TRACK_FACTORS[track_name]
    
    # Number of drivers
    num_drivers = st.slider("👥 Number of Drivers", min_value=2, max_value=20, value=5)
    
    st.markdown("---")
    
    # Show available drivers and teams
    try:
        driver_list = api.get_driver_list()
        team_list = api.get_team_list()
        
        st.markdown("**Available Drivers:**")
        st.write(", ".join(driver_list[:10]) + ("..." if len(driver_list) > 10 else ""))
        
        st.markdown("**Available Teams:**")
        st.write(", ".join(team_list))
    except Exception as e:
        st.error(f"Error loading driver/team data: {str(e)}")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🏎️ Driver Setup")
    
    # Initialize session state for form validation
    if 'form_errors' not in st.session_state:
        st.session_state.form_errors = []
    
    # Create input form
    with st.form("race_input_form"):
        drivers = []
        teams = []
        qualifying_positions = []
        qualifying_times = []
        
        # Get lists for validation
        try:
            available_drivers = api.get_driver_list()
            available_teams = api.get_team_list()
        except Exception as e:
            st.error(f"Error loading driver/team lists: {str(e)}")
            st.stop()
        
        # Track used qualifying positions to prevent duplicates
        used_positions = set()
        form_valid = True
        
        # Create driver inputs in a more organized way
        for i in range(num_drivers):
            st.markdown(f"**Driver {i+1}**")
            cols = st.columns(4)
            
            with cols[0]:
                driver = st.selectbox(
                    "Driver", 
                    available_drivers, 
                    key=f"driver_{i}",
                    help="Select the racing driver"
                )
                
            with cols[1]:
                team = st.selectbox(
                    "Team", 
                    available_teams, 
                    key=f"team_{i}",
                    help="Select the team/constructor"
                )
                
            with cols[2]:
                qual_pos = st.number_input(
                    "Qualifying Position", 
                    min_value=1, 
                    max_value=20, 
                    value=i+1,
                    key=f"qual_pos_{i}",
                    help="Starting grid position"
                )
                
                # Check for duplicate qualifying positions
                if qual_pos in used_positions:
                    st.error(f"Position {qual_pos} is already used! Each driver needs a unique qualifying position.")
                    form_valid = False
                else:
                    used_positions.add(qual_pos)

            with cols[3]:
                # Calculate default time based on position (realistic F1 qualifying times)
                base_time = 82.0  # Base time in seconds (around 1:22)
                position_penalty = (qual_pos - 1) * 0.2  # Each position back adds 0.2 seconds
                default_time_seconds = base_time + position_penalty
                default_time_formatted = f"{int(default_time_seconds // 60)}:{default_time_seconds % 60:06.3f}"
                
                qual_time_str = st.text_input(
                    "Qualifying Time (M:SS.sss)", 
                    value=default_time_formatted, 
                    key=f"qual_time_{i}",
                    help="Enter qualifying time in format M:SS.sss (e.g., 1:22.456)"
                )
                
                qual_time_sec = convert_time_to_seconds(qual_time_str)
                
                if qual_time_sec is None:
                    st.error(f"Invalid qualifying time format for Driver {i+1}")
                    form_valid = False
            
            # Append to lists
            drivers.append(driver)
            teams.append(team)
            qualifying_positions.append(qual_pos)
            qualifying_times.append(qual_time_sec if qual_time_sec is not None else 0)
            
            st.markdown("---")
        
        # Additional validation
        if len(set(drivers)) != len(drivers):
            st.error("Each driver can only be selected once!")
            form_valid = False
        
        # Submit button
        submitted = st.form_submit_button("🚀 Predict Winner & Analyze", use_container_width=True)
        
        # Show validation status
        if not form_valid and submitted:
            st.error("Please fix the errors above before submitting.")

with col2:
    st.subheader("📊 Track Info")
    st.info(f"**Selected Track:** {track_name}")
    
    # Show track factors for top teams
    st.markdown("**Track Advantage:**")
    try:
        top_advantages = sorted(track_factors.items(), key=lambda x: x[1], reverse=True)[:7]
        
        for team, factor in top_advantages:
            advantage = "🟢" if factor > 0.99 else "🔴" if factor < 0.96 else "⚪"
            st.write(f"{advantage} {team}: {factor:.3f}")
    except Exception as e:
        st.error(f"Error displaying track advantages: {str(e)}")

# Prediction and Analysis Section
if submitted and form_valid:
    st.markdown("---")
    
    with st.spinner("🔄 Analyzing race data and generating predictions..."):
        try:
            # Get predictions and SHAP analysis
            results_df, shap_analysis = api.predict_race_winner(
                drivers=drivers,
                qualifying_positions=qualifying_positions,
                qualifying_times=qualifying_times,
                teams=teams,
                track_factors=track_factors,
                track_name=track_name
            )
            
            # Debug: Print what we received (optional - can be removed in production)
            if st.checkbox("Show Debug Info", value=False):
                st.write("🔍 Debug Info:")
                st.write(f"Results DataFrame shape: {results_df.shape if not results_df.empty else 'Empty'}")
                st.write(f"SHAP Analysis type: {type(shap_analysis)}")
                if shap_analysis:
                    st.write(f"SHAP Analysis keys: {list(shap_analysis.keys())}")
            
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.info("Please check your inputs and try again.")
            if st.checkbox("Show Error Details"):
                st.exception(e)  # Show full traceback for debugging
            results_df = pd.DataFrame()
            shap_analysis = None
    
    if not results_df.empty:
        # Winner Announcement
        winner = results_df.iloc[0]
        st.markdown(f"""
        <div class="winner-box">
            <h2>🏆 Predicted Winner: {winner['Driver']}</h2>
            <h3>Team: {winner['Team']} | Win Probability: {winner['WinProbability']:.1%}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Results Table
        st.subheader("📋 Full Race Predictions")
        display_cols = ["Driver", "Team", "QualifyingPosition", "WinProbability", "DriverSkill", "TeamPerformance"]
        
        # Ensure all required columns exist
        available_cols = [col for col in display_cols if col in results_df.columns]
        formatted_df = results_df[available_cols].copy()
        
        if "WinProbability" in formatted_df.columns:
            formatted_df["WinProbability"] = formatted_df["WinProbability"].apply(lambda x: f"{x:.1%}")
        
        st.dataframe(
            formatted_df,
            use_container_width=True,
            hide_index=True
        )
        
        # BEFORE using SHAP analysis, validate it comprehensively
        shap_valid = False
        validation_msg = ""
        
        if shap_analysis and isinstance(shap_analysis, dict):
            shap_valid, validation_msg = validate_shap_data(shap_analysis)
            
            # Check if any valid components exist
            valid_components = [
                key for key in ['winner_analysis', 'feature_importance_fig', 'summary_fig', 'heatmap_fig']
                if key in shap_analysis and shap_analysis[key] is not None
            ]
            
            if valid_components:
                st.header("🔍 SHAP Analysis - What Drives the Predictions?")
                
                # Create tabs for different analyses
                tab1, tab2, tab3, tab4 = st.tabs(["🎯 Winner Analysis", "📊 Feature Importance", "🔥 Feature Heatmap", "📈 Summary Plot"])
                
                with tab1:
                    st.subheader(f"🏆 Detailed Analysis for {winner['Driver']}")
                    
                    winner_analysis = shap_analysis.get('winner_analysis', {})
                    
                    if winner_analysis and isinstance(winner_analysis, dict):
                        # Winner stats
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Win Probability", f"{winner_analysis.get('win_probability', 0):.1%}")
                        with col2:
                            st.metric("Team", winner_analysis.get('team', 'Unknown'))
                        with col3:
                            st.metric("Driver", winner_analysis.get('driver', 'Unknown'))
                        
                        # SHAP analysis plot for winner
                        winner_fig = winner_analysis.get('figure')
                        if winner_fig:
                            st.subheader("📊 SHAP Force Plot for Winner")
                            if not safe_display_figure(winner_fig, "Error displaying winner analysis plot"):
                                st.warning("Could not display winner analysis plot")
                        
                        # Feature impact table
                        analysis_df = winner_analysis.get('analysis_df')
                        if analysis_df is not None and not analysis_df.empty:
                            st.subheader("📋 Feature Impact Breakdown")
                            try:
                                impact_df = analysis_df[['Feature', 'SHAP_Value', 'Feature_Value']].copy()
                                impact_df['Impact'] = impact_df['SHAP_Value'].apply(
                                    lambda x: "🟢 Positive" if x > 0 else "🔴 Negative"
                                )
                                impact_df['SHAP_Value'] = impact_df['SHAP_Value'].apply(lambda x: f"{x:.4f}")
                                st.dataframe(impact_df, use_container_width=True, hide_index=True)
                            except Exception as e:
                                st.error(f"Error displaying feature impact: {str(e)}")
                        else:
                            st.warning("Feature impact data not available")
                    else:
                        st.warning("Winner analysis data not available")
                
                with tab2:
                    st.subheader("📊 Overall Feature Importance")
                    feature_importance_fig = shap_analysis.get('feature_importance_fig')
                    
                    if feature_importance_fig:
                        if safe_display_figure(feature_importance_fig, "Error displaying feature importance plot"):
                            st.markdown("""
                            **Understanding Feature Importance:**
                            - Higher bars indicate features that have more impact on predictions
                            - This shows which factors matter most across all drivers
                            - Values represent average absolute impact on win probability
                            """)
                        else:
                            st.warning("Feature importance plot could not be displayed")
                    else:
                        st.warning("Feature importance plot not available")
                
                with tab3:
                    st.subheader("🔥 Feature Correlation Heatmap")
                    heatmap_fig = shap_analysis.get('heatmap_fig')
                    
                    if heatmap_fig:
                        if safe_display_figure(heatmap_fig, "Error displaying heatmap"):
                            st.markdown("""
                            **Reading the Heatmap:**
                            - Red colors indicate positive correlation
                            - Blue colors indicate negative correlation  
                            - Darker colors show stronger correlations
                            - This helps identify which features work together
                            """)
                        else:
                            st.warning("Correlation heatmap could not be displayed")
                    else:
                        st.warning("Correlation heatmap not available")
                
                with tab4:
                    st.subheader("📈 SHAP Summary Analysis")
                    summary_fig = shap_analysis.get('summary_fig')
                    
                    if summary_fig:
                        if not safe_display_figure(summary_fig, "Error displaying summary plot"):
                            st.warning("Summary plot could not be displayed")
                    else:
                        # Alternative summary if plot fails
                        st.write("📊 Feature Impact Summary:")
                        
                        # Try to create alternative summary
                        shap_values = shap_analysis.get('shap_values')
                        feature_names = shap_analysis.get('feature_names')
                        
                        if shap_values is not None and feature_names is not None:
                            try:
                                feature_impact = np.abs(shap_values).mean(axis=0)
                                summary_df = pd.DataFrame({
                                    'Feature': feature_names,
                                    'Average Impact': feature_impact
                                }).sort_values('Average Impact', ascending=False)
                                
                                st.bar_chart(summary_df.set_index('Feature')['Average Impact'])
                            except Exception as e:
                                st.error(f"Error creating alternative summary: {str(e)}")
                        else:
                            st.warning("SHAP summary data not available")
            else:
                st.warning("SHAP analysis generated but no valid components found")
                st.info("Showing basic predictions with fallback analysis")
                
                # Create and show fallback analysis
                fallback_analysis = create_fallback_analysis(results_df, api)
                if fallback_analysis:
                    st.subheader("📊 Basic Analysis")
                    
                    if 'basic_importance' in fallback_analysis:
                        st.write("**Feature Importance (from model):**")
                        st.dataframe(fallback_analysis['basic_importance'])
                    
                    if 'probability_range' in fallback_analysis:
                        prob_range = fallback_analysis['probability_range']
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Min Probability", f"{prob_range['min']:.1%}")
                        with col2:
                            st.metric("Average Probability", f"{prob_range['mean']:.1%}")
                        with col3:
                            st.metric("Max Probability", f"{prob_range['max']:.1%}")
        else:
            st.warning("SHAP analysis failed to generate or is invalid")
            st.info("Showing basic predictions with fallback analysis")
            
            # Create and show fallback analysis
            fallback_analysis = create_fallback_analysis(results_df, api)
            if fallback_analysis:
                st.subheader("📊 Basic Analysis")
                
                if 'basic_importance' in fallback_analysis:
                    st.write("**Feature Importance (from model):**")
                    st.dataframe(fallback_analysis['basic_importance'])
        
        # Additional Insights (always show)
        st.subheader("💡 Key Insights")
        
        insights_col1, insights_col2 = st.columns(2)
        
        with insights_col1:
            st.markdown("**🎯 Prediction Factors:**")
            st.write(f"• **Qualifying Position**: P{winner.get('QualifyingPosition', 'N/A')} start")
            
            driver_skill = winner.get('DriverSkill', 'N/A')
            if isinstance(driver_skill, (int, float)):
                st.write(f"• **Driver Skill**: {driver_skill:.1f}/100")
            else:
                st.write(f"• **Driver Skill**: {driver_skill}")
            
            team_performance = winner.get('TeamPerformance', 'N/A')
            if isinstance(team_performance, (int, float)):
                st.write(f"• **Team Performance**: {team_performance:.1f}/100")
            else:
                st.write(f"• **Team Performance**: {team_performance}")
                
            st.write(f"• **Track Advantage**: {track_factors.get(winner['Team'], 1.0):.3f}")
        
        with insights_col2:
            st.markdown("**📊 Race Statistics:**")
            try:
                avg_prob = results_df['WinProbability'].mean()
                st.write(f"• **Average Win Probability**: {avg_prob:.1%}")
                
                top_3_probs = results_df.head(3)['WinProbability']
                st.write(f"• **Top 3 Probability Range**: {top_3_probs.min():.1%} - {top_3_probs.max():.1%}")
                
                last_driver = results_df.iloc[-1]
                st.write(f"• **Biggest Upset Potential**: {last_driver['Driver']} ({last_driver['WinProbability']:.1%})")
            except Exception as e:
                st.error(f"Error calculating race statistics: {str(e)}")
    
    elif submitted and form_valid:
        st.error("❌ Prediction failed. Please check your inputs and try again.")

# Memory cleanup (run at end of script)
try:
    # Close all matplotlib figures to prevent memory leaks
    plt.close('all')
    print("✅ Memory cleanup completed")
except Exception as cleanup_error:
    print(f"⚠️ Cleanup warning: {cleanup_error}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; margin-top: 2rem;'>
    <p>🏎️ F1 Race Predictor with AI-Powered Analysis | Built with Streamlit & SHAP</p>
    <p><em>Predictions are for entertainment purposes only</em></p>
</div>
""", unsafe_allow_html=True)