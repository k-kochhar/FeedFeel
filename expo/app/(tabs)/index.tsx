import React, { useState } from 'react';
import { StyleSheet, TouchableOpacity, Platform, TextInput, Vibration, View } from 'react-native';
import * as Haptics from 'expo-haptics';
import { ThemedView } from '@/components/ThemedView';
import { ThemedText } from '@/components/ThemedText';

export default function HomeScreen() {
  const [inputText, setInputText] = useState('');
  const [isVibrating, setIsVibrating] = useState(false);

  const handleTextToVibration = () => {
    if (isVibrating || !inputText) return;
    setIsVibrating(true);

    // Create vibration pattern based on text
    const pattern = createVibrationPattern(inputText);
    
    if (Platform.OS === 'ios') {
      // iOS: Use Haptics for a more nuanced experience
      playHapticSequence(pattern);
    } else {
      // Android: Use Vibration API
      Vibration.vibrate(pattern, false);
    }

    // Ensure vibration stops after 5 seconds max
    setTimeout(() => {
      Vibration.cancel();
      setIsVibrating(false);
    }, 5000);
  };

  // Convert text to vibration pattern
  const createVibrationPattern = (text: string): number[] => {
    // Simple algorithm:
    // - Short vibration (100ms) for lowercase letters
    // - Medium vibration (200ms) for uppercase letters
    // - Long vibration (300ms) for numbers
    // - Pause (100ms) for spaces and punctuation
    // - Max 5 seconds total
    
    const pattern: number[] = [];
    let totalDuration = 0;
    const MAX_DURATION = 5000; // 5 seconds max
    
    for (let i = 0; i < text.length; i++) {
      const char = text[i];
      let duration = 0;
      
      if (/[a-z]/.test(char)) {
        duration = 100; // lowercase
      } else if (/[A-Z]/.test(char)) {
        duration = 200; // uppercase
      } else if (/[0-9]/.test(char)) {
        duration = 300; // numbers
      } else {
        duration = 100; // spaces and punctuation (pause)
      }
      
      // Add pause between vibrations
      if (pattern.length > 0) {
        pattern.push(50);
        totalDuration += 50;
      }
      
      pattern.push(duration);
      totalDuration += duration;
      
      if (totalDuration >= MAX_DURATION) break;
    }
    
    return pattern;
  };

  // For iOS, simulate pattern with Haptics
  const playHapticSequence = async (pattern: number[]) => {
    for (let i = 0; i < pattern.length; i++) {
      if (i % 2 === 0) { // Vibration
        const duration = pattern[i];
        if (duration > 250) {
          await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Heavy);
        } else if (duration > 150) {
          await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
        } else {
          await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
        }
      }
      
      // Wait for the pattern duration
      await new Promise(resolve => setTimeout(resolve, pattern[i]));
    }
  };

  return (
    <ThemedView style={styles.container}>
      <ThemedText style={styles.title}>Text-to-Vibration</ThemedText>
      
      <View style={styles.inputContainer}>
        <TextInput
          style={styles.textInput}
          placeholder="Enter text to vibrate..."
          value={inputText}
          onChangeText={setInputText}
          maxLength={50}
        />
      </View>
      
      <TouchableOpacity 
        style={[
          styles.button,
          isVibrating && styles.buttonDisabled,
          !inputText && styles.buttonDisabled
        ]}
        onPress={handleTextToVibration}
        disabled={isVibrating || !inputText}
      >
        <ThemedText style={styles.buttonText}>
          {isVibrating ? "VIBRATING..." : "VIBRATE"}
        </ThemedText>
      </TouchableOpacity>
      
      <ThemedText style={styles.helpText}>
        Lowercase = light vibration
        {'\n'}Uppercase = medium vibration
        {'\n'}Numbers = strong vibration
        {'\n'}(Max 5 seconds)
      </ThemedText>
    </ThemedView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 16,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 30,
  },
  inputContainer: {
    width: '100%',
    marginBottom: 24,
  },
  textInput: {
    backgroundColor: 'white',
    borderRadius: 8,
    padding: 12,
    fontSize: 16,
    borderWidth: 1,
    borderColor: '#ddd',
    width: '100%',
  },
  button: {
    backgroundColor: '#A1CEDC',
    paddingVertical: 16,
    paddingHorizontal: 40,
    borderRadius: 12,
    elevation: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 3,
  },
  buttonDisabled: {
    backgroundColor: '#ccc',
    opacity: 0.8,
  },
  buttonText: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
  },
  helpText: {
    marginTop: 24,
    textAlign: 'center',
    fontSize: 14,
    opacity: 0.7,
    lineHeight: 20,
  },
});
