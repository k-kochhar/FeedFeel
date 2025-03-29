import React from 'react';
import { View, Button } from 'react-native';
import * as Haptics from 'expo-haptics';

const VibrateButton: React.FC = () => {
  const triggerVibration = async () => {
    try {
      await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    } catch (error) {
      console.error('Haptics error:', error);
    }
  };

  return (
    <View style={{ margin: 20 }}>
      <Button title="Vibrate" onPress={triggerVibration} />
    </View>
  );
};

export default VibrateButton;
