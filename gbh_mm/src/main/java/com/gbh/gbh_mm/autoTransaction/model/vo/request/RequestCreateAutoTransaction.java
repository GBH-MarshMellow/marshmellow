package com.gbh.gbh_mm.autoTransaction.model.vo.request;

import lombok.Getter;

@Getter
public class RequestCreateAutoTransaction {
    private String withdrawalAccountNo;
    private String depositAccountNo;
    private String dueDate;
    private long wishListPk;
    private int transactionBalance;
}
