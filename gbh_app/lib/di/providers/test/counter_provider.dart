// lib/di/providers/counter_provider.dart
import 'package:flutter_riverpod/flutter_riverpod.dart';

// 간단한 정수 상태를 관리하는 Provider (초기값 0)
final counterProvider = StateProvider<int>((ref) => 0);