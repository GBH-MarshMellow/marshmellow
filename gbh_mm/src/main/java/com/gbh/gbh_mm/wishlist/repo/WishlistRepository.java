package com.gbh.gbh_mm.wishlist.repo;

import com.gbh.gbh_mm.wishlist.model.entity.Wishlist;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface WishlistRepository extends JpaRepository<Wishlist, Long> {
    List<Wishlist> findAllByUser_UserPk(long userPk);

}
