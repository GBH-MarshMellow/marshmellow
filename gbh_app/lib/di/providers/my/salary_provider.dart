import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:marshmellow/data/repositories/my/salary_repository.dart';
import 'package:marshmellow/di/providers/api_providers.dart';

final mySalaryRepositoryProvider = Provider<MySalaryRepository>((ref) {
  final mySalaryApi = ref.watch(mySalaryApiProvider);
  return MySalaryRepository(mySalaryApi);
});