import { StyleSheet, TouchableOpacity, Platform } from 'react-native';
import * as Haptics from 'expo-haptics';
import { Vibration } from 'react-native';
import { ThemedView } from '@/components/ThemedView';
import { ThemedText } from '@/components/ThemedText';

export default function HomeScreen() {
  const handleVibrate = () => {
    if (Platform.OS === 'android') {
      // Android: vibrate for 5 seconds
      Vibration.vibrate(5000);
    } 
  };

  return (
    <ThemedView style={styles.container}>
      <TouchableOpacity 
        style={styles.button}
        onPress={handleVibrate}
      >
        <ThemedText style={styles.buttonText}>VIBRATE</ThemedText>
      </TouchableOpacity>
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
  button: {
    backgroundColor: '#A1CEDC',
    paddingVertical: 20,
    paddingHorizontal: 60,
    borderRadius: 15,
    elevation: 5,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 3 },
    shadowOpacity: 0.3,
    shadowRadius: 4,
  },
  buttonText: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#333',
  },
});
