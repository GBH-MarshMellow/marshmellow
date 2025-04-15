import 'package:flutter/material.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';

// 위젯
import 'package:marshmellow/presentation/widgets/custom_appbar/custom_appbar.dart';

class QuitInfoPage extends StatelessWidget {
  const QuitInfoPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: CustomAppbar(
        title: '퇴사 망상',
      ),
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Center(
                child: Image.asset(
                  'assets/images/characters/char_angry_notebook.png',
                  width: 150,
                  height: 150,
                ),
              ),
              const SizedBox(height: 50),
              _buildQuestionSection(
                question: 'Q1. 퇴사하기 좋은 날은?',
                subtitle: '주휴수당을 받을 수 있는 날로 퇴사일을 정하세요.',
                content1:
                    '퇴사 시기를 정할 때는 주휴수당을 고려하는 것이 좋습니다. 주휴수당은 일주일 동안 정해진 근무일을 성실히 이행한 근로자에게 지급되는 유급휴일 수당입니다. 주 15시간 이상 근무하는 경우, 평균적으로 주 1회 이상의 유급휴일이 발생하며, 주 5일 근무 기준으로는 \'8시간 × 시급\' 금액을 주휴수당으로 받게 됩니다.\n\n'
                    '예를 들어, 월요일에 퇴사하면 그 전 주 일요일까지 근로관계가 유지되므로 해당 주의 주휴수당을 받을 수 있습니다. 반면 금요일에 퇴사하면 주말까지의 근로관계가 인정되지 않아 주휴수당을 받을 수 없습니다. 실제 근무일 수는 비슷할 수 있지만, 퇴사일을 어떻게 설정하느냐에 따라 월급에 차이가 생길 수 있습니다. 이왕이면 주휴수당을 받을 수 있는 월요일 퇴사가 경제적으로 유리합니다.',
              ),
              const SizedBox(height: 50),
              _buildQuestionSection(
                question: 'Q2. 퇴사 전 비상금은 얼마나?',
                subtitle: '최소 3개월 치 생활비를 모아두세요.',
                content1:
                    '직장 생활자의 경우는 3개월 치 생활비, 프리랜서나 자영업자의 경우는 6개월치 생활비를 적정 비상금으로 봅니다. 재취업이나 다른 소득 활동을 시작하게 될 때까지 필요한 시간을 그 정도로 보는 건데요. 무슨 일이 언제 발생할지도 모르는 인생에서 적정규모의 비상자금을 보유하는 것은 중요한 삶의 안전관리죠.\n\n'
                    '자의든 타의든 직장을 나오게 된다면, 비상자금은 심신의 안정을 돕습니다. 비상자금은 급할 때 꺼내 쓸 수 없는 곳에 묶어놓기보다 언제든 꺼내 쓸 수 있는 금융상품으로 운영하는 것이 바람직합니다.\n\n'
                    '미처 모아 둔 비상자금이 없을 때는 퇴직금이 역할을 대신합니다. 따라서 퇴사하기 전에 자신의 퇴직금 예상 수령액이 어느 정도 되는지부터 먼저 알아볼 필요가 있죠. 여기에 실업급여까지 합쳐지면 다음 일자리를 구할 때까지 기본 생활을 영위할 수 있을 겁니다.\n\n'
                    '다만 퇴사 후 여행이나, 새로운 걸 배우거나 할 경우엔 기본 생활비 외에도 추가 비용이 들어가게 됩니다. 이런 부분까지 고려한다면 역시 비상자금을 갖고 있는 것은 매우 유용하겠죠. 비상자금은 얼마를 보유하느냐의 문제라기보다 돈을 벌 수 없는 기간을 대비하는 일종의 \'안전관리 습관\'으로 보시는 것이 좋습니다.',
              ),
              const SizedBox(height: 50),
              _buildQuestionSection(
                question: 'Q3. 쉬는 동안 생활비 분배는?',
                subtitle: '나의 경제생활을 과거, 현재, 미래로 나눠보세요',
                content1:
                    '균형 잡힌 분배를 위해서는 돈이 빠져나가는 3가지 경로를 알아야 합니다. ①대출상환 ②현재지출 ③미래저축. 대출은 과거에 미리 빌려 쓴 돈을 현재 갚고 있는 거니까 과거형의 경제생활이죠. 지금 먹고사는 문제는 현재형, 나중을 위해 지금의 여력을 일부 떼어 두는 저축은 미래형입니다. 이렇게 나의 경제생활은 과거, 현재, 미래가 공존하고 있습니다.\n\n'
                    '수입이 끊기고 나면 보유한 비상자금 액수를 놓고 월별 가용자금이 얼마인지부터 재빠르게 산출해야 하는데요.\n',
                contentTitle2: '① 가장 먼저 계산해야 할 것은 고정지출\n',
                content2:
                    '피도 눈물도 없이 빠져나가는 고정비용부터 우선적으로 집계합니다. 그 비용을 제외하고 나머지로 생활비를 책정해야겠죠. 월세, 관리비, 공과금, 각종 렌탈비, 인터넷, 휴대전화, 보장성 보험료, 각종 할부금, 대출상환 등이 고정지출에 해당됩니다. 자동차를 보유하고 있는 경우에는 보험금이나 세금부터 얼마인지 언제 납부해야 하는지를 챙겨야 목돈 나갈 때 당황하지 않을 수 있습니다.\n',
                contentTitle3: '② 놀면 더 쓴다. 최저생계비 책정하기\n',
                content3:
                    '삶의 아이러니는 일할 때보다 놀 때 돈이 더 든다는 데 있습니다. 일에 쫓겨 사느라 그동안 못하고 살았던 거 해보는건 당연한 수순이겠죠. 다만 이것저것 우선순위 없이 내키는 대로 하다가는 정작 중요한 상황에 돈이 부족해지는 결과를 초래할 수 있으니 주어진 여력을 사전에 잘 배분해 놓는 게 중요하단 얘기죠.\n',
                contentTitle4: '③ 추가 소비여력은 잠시 접어두기\n',
                content4:
                    '신용카드 얘깁니다. 우리나라 경제활동 인구 1인당 3.9개의 신용카드를 보유하고 있다고 합니다. 비상상황이니만큼 정해진 예산 안에서 돈을 쓰고 신용카드나 마이너스통장과 같은 추가 소비여력은 잠시 서랍에 넣어두시고 생활할 필요가 있습니다.',
              ),
              const SizedBox(height: 50),
              _buildQuestionSection(
                question: 'Q4. 퇴사 준비 시 챙겨야 할 서류는?',
                subtitle: '퇴사 준비 시 챙겨야 할 서류 목록',
                contentTitle1: '① 경력증명서(재직증명서)',
                contentTitle2: '② 근로소득원천징수영수증',
                contentTitle3: '③ 급여명세서',
                contentTitle4: '④ 퇴직금정산내역서',
                contentTitle5: '⑤ 사직서 사본\n\n',
                content5:
                    '퇴사 후 전 직장에 연락할 일은 최대한 피하고 싶은 게 사람 마음일 것입니다. 하지만 연말정산 시 필요한 서류를 발급받거나, 이직 과정에서 경력 확인을 위해 관련 자료가 필요한 경우에는 어쩔 수 없이 전 직장에 연락해야 할 수도 있습니다. 이러한 서류들은 퇴사 이후에 요청하면 발급 절차가 번거로울 수 있으므로, 퇴사 전에 미리 준비해두는 것이 효율적입니다.',
              ),
            ],
          ),
        ),
      ),
    );
  }

  // 질문과 내용을 보여주는 위젯
  Widget _buildQuestionSection({
    required String question,
    required String subtitle,
    String? contentTitle1,
    String? content1,
    String? contentTitle2,
    String? content2,
    String? contentTitle3,
    String? content3,
    String? contentTitle4,
    String? content4,
    String? contentTitle5,
    String? content5,
  }) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          question,
          style: AppTextStyles.bodyLarge.copyWith(
            fontWeight: FontWeight.bold,
            color: AppColors.textPrimary,
          ),
        ),
        const SizedBox(height: 20),
        Text(
          subtitle,
          style: AppTextStyles.bodyMedium.copyWith(
            color: AppColors.textPrimary,
          ),
        ),
        const SizedBox(height: 10),
        if (contentTitle1 != null)
          Text(
            contentTitle1,
            style: AppTextStyles.bodySmall.copyWith(
              fontWeight: FontWeight.w400,
              color: AppColors.textPrimary,
            ),
          ),
        if (content1 != null)
          Text(
            content1,
            style: AppTextStyles.bodySmall.copyWith(
              fontWeight: FontWeight.w300,
              color: AppColors.textPrimary,
              height: 1.6,
            ),
          ),
        const SizedBox(height: 10),
        if (contentTitle2 != null)
          Text(
            contentTitle2,
            style: AppTextStyles.bodySmall.copyWith(
              fontWeight: FontWeight.w400,
              color: AppColors.textPrimary,
            ),
          ),
        if (content2 != null)
          Text(
            content2,
            style: AppTextStyles.bodySmall.copyWith(
              fontWeight: FontWeight.w300,
              color: AppColors.textPrimary,
              height: 1.6,
            ),
          ),
        const SizedBox(height: 10),
        if (contentTitle3 != null)
          Text(
            contentTitle3,
            style: AppTextStyles.bodySmall.copyWith(
              fontWeight: FontWeight.w400,
              color: AppColors.textPrimary,
            ),
          ),
        if (content3 != null)
          Text(
            content3,
            style: AppTextStyles.bodySmall.copyWith(
              fontWeight: FontWeight.w300,
              color: AppColors.textPrimary,
              height: 1.6,
            ),
          ),
        const SizedBox(height: 10),
        if (contentTitle4 != null)
          Text(
            contentTitle4,
            style: AppTextStyles.bodySmall.copyWith(
              fontWeight: FontWeight.w400,
              color: AppColors.textPrimary,
            ),
          ),
        if (content4 != null)
          Text(
            content4,
            style: AppTextStyles.bodySmall.copyWith(
              fontWeight: FontWeight.w300,
              color: AppColors.textPrimary,
              height: 1.6,
            ),
          ),
        const SizedBox(height: 10),
        if (contentTitle5 != null)
          Text(
            contentTitle5,
            style: AppTextStyles.bodySmall.copyWith(
              fontWeight: FontWeight.w400,
              color: AppColors.textPrimary,
            ),
          ),
        if (content5 != null)
          Text(
            content5,
            style: AppTextStyles.bodySmall.copyWith(
              fontWeight: FontWeight.w300,
              color: AppColors.textPrimary,
              height: 1.6,
            ),
          ),
      ],
    );
  }
}
