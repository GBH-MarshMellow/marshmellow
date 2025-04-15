// lib/data/models/my/user_detail_info.dart
class UserDetailInfo {
  final String? salaryAccount;
  final int? salaryAmount;
  final int? salaryDate;
  final String? budgetFeature;
  final String? budgetAlarmTime;
  final String? userKeyYn;

  UserDetailInfo({
    this.salaryAccount,
    this.salaryAmount,
    this.salaryDate,
    this.budgetFeature,
    this.budgetAlarmTime,
    this.userKeyYn,
  });

factory UserDetailInfo.fromJson(Map<String, dynamic> json) {
  return UserDetailInfo(
    salaryAccount: json['salaryAccount'] as String?,
    salaryAmount: json['salaryAmount'] is String ? 
                  int.tryParse(json['salaryAmount']) : 
                  json['salaryAmount'] as int?,
    salaryDate: json['salaryDate'] is String ? 
                int.tryParse(json['salaryDate']) : 
                json['salaryDate'] as int?,
    budgetFeature: json['budgetFeature'] as String?,
    budgetAlarmTime: json['budgetAlarmTime'] as String?,
    userKeyYn: json['userKeyYn'] as String?,
  );
}

  factory UserDetailInfo.empty() {
    return UserDetailInfo(
      salaryAccount: null,
      salaryAmount: null,
      salaryDate: null,
      budgetFeature: null,
      budgetAlarmTime: null,
      userKeyYn: null,
    );
  }
}