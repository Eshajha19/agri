"""
Test suite for Voice Assistant
Tests cover: voice processing, intent detection, language support, offline functionality
"""

import pytest
import asyncio
from voice_assistant import (
    VoiceAssistant,
    OfflineLanguageModel,
    VoiceInput,
    VoiceQueryAnalyzer,
    detect_language,
    SUPPORTED_LANGUAGES,
    INTENT_PATTERNS,
)


class TestLanguageDetection:
    """Test language detection from text"""
    
    def test_hindi_detection(self):
        """Test Hindi text detection"""
        text = "मेरी फसल को क्या समस्या है?"
        lang = detect_language(text)
        assert lang == "hi"
    
    def test_english_detection(self):
        """Test English text detection"""
        text = "What problem does my crop have?"
        lang = detect_language(text)
        assert lang == "en"
    
    def test_gujarati_detection(self):
        """Test Gujarati text detection"""
        text = "મારી પાક માટે શું સમસ્યા છે?"
        lang = detect_language(text)
        assert lang == "gu"
    
    def test_tamil_detection(self):
        """Test Tamil text detection"""
        text = "என் சாகுபடிக்கு என்ன சிக்கல் உள்ளது?"
        lang = detect_language(text)
        assert lang == "ta"


class TestIntentDetection:
    """Test intent detection from queries"""
    
    def setup_method(self):
        """Setup for each test"""
        self.language_model = OfflineLanguageModel()
    
    def test_crop_health_intent_hindi(self):
        """Test crop health intent in Hindi"""
        query = "मेरी फसल को क्या समस्या है?"
        intent, confidence = self.language_model.detect_intent(query)
        assert intent == "crop_health"
        assert confidence > 0.5
    
    def test_crop_health_intent_english(self):
        """Test crop health intent in English"""
        query = "What problem does my crop have?"
        intent, confidence = self.language_model.detect_intent(query)
        assert intent == "crop_health"
        assert confidence > 0.5
    
    def test_weather_alert_intent(self):
        """Test weather alert intent"""
        query = "मौसम कैसा रहेगा?"
        intent, confidence = self.language_model.detect_intent(query)
        assert intent == "weather_alert"
        assert confidence > 0.5
    
    def test_fertilizer_guide_intent(self):
        """Test fertilizer guide intent"""
        query = "गेहूँ को कौन सी खाद दें?"
        intent, confidence = self.language_model.detect_intent(query)
        assert intent == "fertilizer_guide"
        assert confidence > 0.5
    
    def test_irrigation_advice_intent(self):
        """Test irrigation advice intent"""
        query = "सिंचाई कब करें?"
        intent, confidence = self.language_model.detect_intent(query)
        assert intent == "irrigation_advice"
        assert confidence > 0.5
    
    def test_pest_management_intent(self):
        """Test pest management intent"""
        query = "कीड़ों से कैसे बचें?"
        intent, confidence = self.language_model.detect_intent(query)
        assert intent == "pest_management"
        assert confidence > 0.5
    
    def test_yield_prediction_intent(self):
        """Test yield prediction intent"""
        query = "इस बार पैदावार कितनी होगी?"
        intent, confidence = self.language_model.detect_intent(query)
        assert intent == "yield_prediction"
        assert confidence > 0.5


class TestEntityExtraction:
    """Test entity extraction from queries"""
    
    def setup_method(self):
        """Setup for each test"""
        self.language_model = OfflineLanguageModel()
    
    def test_extract_crop_entity(self):
        """Test crop entity extraction"""
        query = "मेरे चावल को समस्या है"
        entities = self.language_model.extract_entities(query, "crop_health")
        assert "crop" in entities or "chawal" in query.lower()
    
    def test_extract_disease_entity(self):
        """Test disease entity extraction"""
        query = "मेरी फसल को fungal infection है"
        entities = self.language_model.extract_entities(query, "crop_health")
        assert "disease" in entities


class TestVoiceAssistant:
    """Test main VoiceAssistant class"""
    
    def setup_method(self):
        """Setup for each test"""
        self.assistant = VoiceAssistant(offline_mode=True)
    
    def test_assistant_initialization(self):
        """Test voice assistant initialization"""
        assert self.assistant is not None
        assert self.assistant.offline_mode is True
        assert len(self.assistant.offline_cache) > 0
    
    def test_session_creation(self):
        """Test session creation"""
        session = self.assistant.create_session("test_user_001", "hi")
        assert session.session_id is not None
        assert session.user_id == "test_user_001"
        assert session.language_code == "hi"
        assert session.offline_mode is True
    
    def test_multiple_sessions(self):
        """Test multiple sessions"""
        session1 = self.assistant.create_session("user_1", "hi")
        session2 = self.assistant.create_session("user_2", "en")
        
        assert session1.session_id != session2.session_id
        assert session1.language_code == "hi"
        assert session2.language_code == "en"
    
    def test_session_retrieval(self):
        """Test session history retrieval"""
        session = self.assistant.create_session("test_user", "hi")
        history = self.assistant.get_session_history(session.session_id)
        
        assert history["session_id"] == session.session_id
        assert history["user_id"] == "test_user"
        assert history["language"] == "hi"
    
    def test_process_voice_input(self):
        """Test voice input processing"""
        session = self.assistant.create_session("test_user", "hi")
        
        voice_input = VoiceInput(
            audio_bytes=b"",
            language_code="hi",
            transcript="मेरी फसल को क्या समस्या है?",
        )
        
        response = self.assistant.process_voice_input(voice_input, session.session_id)
        
        assert response.text is not None
        assert len(response.text) > 0
        assert response.language_code == "hi"
        assert response.intent == "crop_health"
    
    def test_offline_cache_populated(self):
        """Test offline cache is populated"""
        cache = self.assistant.offline_cache
        
        assert "crop_diseases" in cache
        assert "fertilizer_recommendations" in cache
        assert "irrigation_schedules" in cache
        assert "weather_alerts" in cache
    
    def test_crop_disease_in_cache(self):
        """Test crop diseases in cache"""
        crops = ["rice", "wheat", "cotton", "maize"]
        for crop in crops:
            assert crop in self.assistant.offline_cache["crop_diseases"]
            assert len(self.assistant.offline_cache["crop_diseases"][crop]) > 0
    
    def test_fertilizer_recommendations_in_cache(self):
        """Test fertilizer recommendations in cache"""
        crops = ["rice", "wheat", "cotton"]
        for crop in crops:
            assert crop in self.assistant.offline_cache["fertilizer_recommendations"]


class TestQueryAnalyzer:
    """Test query analysis"""
    
    def test_analyze_quality_score(self):
        """Test query quality analysis"""
        query = "मेरी फसल को क्या समस्या है?"
        analysis = VoiceQueryAnalyzer.analyze(query, "hi")
        
        assert analysis["query"] == query
        assert analysis["language"] == "hi"
        assert analysis["clarity_score"] >= 0
        assert analysis["clarity_score"] <= 1
    
    def test_analyze_crop_mention_detection(self):
        """Test crop mention detection"""
        query = "मेरे गेहूँ में समस्या है"
        analysis = VoiceQueryAnalyzer.analyze(query, "hi")
        
        # Should detect crop mention
        assert "has_crop_mention" in analysis
    
    def test_analyze_disease_mention_detection(self):
        """Test disease mention detection"""
        query = "मेरी फसल बीमार है"
        analysis = VoiceQueryAnalyzer.analyze(query, "hi")
        
        # Should detect disease mention
        assert "has_disease_mention" in analysis


class TestLanguageSupport:
    """Test language support"""
    
    def test_all_languages_configured(self):
        """Test all languages are properly configured"""
        for code in SUPPORTED_LANGUAGES.keys():
            assert "name" in SUPPORTED_LANGUAGES[code]
            assert "label" in SUPPORTED_LANGUAGES[code]
    
    def test_response_templates_exist(self):
        """Test response templates exist for all languages"""
        from voice_assistant import RESPONSE_TEMPLATES
        
        required_keys = ["crop_health", "weather_alert", "fertilizer", "irrigation", "greeting", "error"]
        
        for lang_code in ["hi", "en", "mr"]:
            if lang_code in RESPONSE_TEMPLATES:
                for key in required_keys:
                    assert key in RESPONSE_TEMPLATES[lang_code]
    
    def test_intent_patterns_exist(self):
        """Test intent patterns exist"""
        required_intents = [
            "crop_health",
            "weather_alert",
            "fertilizer_guide",
            "irrigation_advice",
            "yield_prediction",
            "pest_management",
        ]
        
        for intent in required_intents:
            assert intent in INTENT_PATTERNS
            assert len(INTENT_PATTERNS[intent]) > 0


class TestOfflineMode:
    """Test offline mode functionality"""
    
    def test_offline_mode_initialization(self):
        """Test offline mode initialization"""
        assistant = VoiceAssistant(offline_mode=True)
        assert assistant.offline_mode is True
    
    def test_offline_response_generation(self):
        """Test response generation in offline mode"""
        assistant = VoiceAssistant(offline_mode=True)
        session = assistant.create_session("test_user", "hi")
        
        voice_input = VoiceInput(
            audio_bytes=b"",
            language_code="hi",
            transcript="गेहूँ को खाद कौन सी दें?",
        )
        
        response = assistant.process_voice_input(voice_input, session.session_id)
        
        # Should return offline-available response
        assert response.offline_available is True
        assert len(response.text) > 0


class TestErrorHandling:
    """Test error handling"""
    
    def test_invalid_session_error(self):
        """Test error on invalid session"""
        assistant = VoiceAssistant(offline_mode=True)
        
        voice_input = VoiceInput(
            audio_bytes=b"",
            language_code="hi",
            transcript="test",
        )
        
        with pytest.raises(ValueError):
            assistant.process_voice_input(voice_input, "invalid_session_id")
    
    def test_invalid_language_code(self):
        """Test handling of unsupported language"""
        assistant = VoiceAssistant(offline_mode=True)
        # Should still work with fallback
        session = assistant.create_session("test_user", "xx")
        assert session is not None


# Integration Tests
class TestIntegration:
    """Integration tests for full workflows"""
    
    def test_complete_workflow_hindi(self):
        """Test complete workflow in Hindi"""
        assistant = VoiceAssistant(offline_mode=True)
        
        # Step 1: Create session
        session = assistant.create_session("test_user", "hi")
        assert session is not None
        
        # Step 2: Process query
        voice_input = VoiceInput(
            audio_bytes=b"",
            language_code="hi",
            transcript="मेरी फसल को क्या समस्या है?",
        )
        response = assistant.process_voice_input(voice_input, session.session_id)
        
        # Step 3: Verify response
        assert response.text is not None
        assert response.language_code == "hi"
        assert response.intent == "crop_health"
        
        # Step 4: Get session history
        history = assistant.get_session_history(session.session_id)
        assert history["last_query"] == "मेरी फसल को क्या समस्या है?"
    
    def test_complete_workflow_english(self):
        """Test complete workflow in English"""
        assistant = VoiceAssistant(offline_mode=True)
        
        session = assistant.create_session("test_user", "en")
        
        voice_input = VoiceInput(
            audio_bytes=b"",
            language_code="en",
            transcript="Which fertilizer for wheat?",
        )
        response = assistant.process_voice_input(voice_input, session.session_id)
        
        assert response.text is not None
        assert response.language_code == "en"
        assert response.intent == "fertilizer_guide"
    
    def test_multiple_queries_same_session(self):
        """Test multiple queries in same session"""
        assistant = VoiceAssistant(offline_mode=True)
        session = assistant.create_session("test_user", "hi")
        
        queries = [
            "मेरी फसल को क्या समस्या है?",
            "गेहूँ को खाद कौन सी दें?",
            "सिंचाई कब करें?",
        ]
        
        for query in queries:
            voice_input = VoiceInput(
                audio_bytes=b"",
                language_code="hi",
                transcript=query,
            )
            response = assistant.process_voice_input(voice_input, session.session_id)
            assert response.text is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
