// di/providers/lifecycle_provider.dart
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../core/utils/lifecycle/app_lifecycle_manager.dart';

final appLifecycleManagerProvider = Provider.autoDispose<AppLifecycleManager>((ref) {
  final manager = AppLifecycleManager(ref);
  
  ref.onDispose(() {
    manager.dispose();
  });
  
  return manager;
});